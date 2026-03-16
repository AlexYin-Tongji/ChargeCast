"""Recursive endogenous multi-step evaluation.

Evaluation protocol:
1. Start from real history window (length=SEQ_LEN).
2. Roll out `--steps` recursively:
   - endogenous nodes use model predictions from previous step,
   - exogenous nodes are forced with ground-truth values at every step.
3. After one rollout block finishes, history is reset using real observations,
   then start next block.

Outputs:
- flowEvaluation/results/recursive_endo_metrics.csv
- flowEvaluation/results/recursive_endo_metrics.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.alignment import align_flow_to_nodes, crop_adjacency
from src.data.flow_data import FlowDataLoader
from src.models.flow_predictor import FlowPredictor
from src.models.topology import TopologyProcessor


def load_model_for_eval(device: torch.device) -> Tuple[FlowPredictor, Dict]:
    ckpt_path = os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    topo_order = ckpt["topo_order"]
    canonical_nodes = ckpt.get("canonical_nodes", topo_order)
    exogenous_nodes = ckpt["exogenous_nodes"]
    upstream_dict = ckpt.get("upstream_dict", {})

    if not upstream_dict:
        loader = FlowDataLoader(config.DATA_DIR)
        adj = loader.load_adjacency("adj.csv")
        try:
            adj = crop_adjacency(adj, canonical_nodes)
        except KeyError as exc:
            raise ValueError(
                "Checkpoint nodes are incompatible with current adj.csv. Please retrain first."
            ) from exc
        topo = TopologyProcessor(adj)
        upstream_dict = {node: topo.get_upstream_nodes(node) for node in topo_order}

    model = FlowPredictor(
        num_nodes=len(topo_order),
        node_ids=topo_order,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=upstream_dict,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise ValueError("Checkpoint missing norm_stats.")
    model.set_normalization(norm_stats["mean"], norm_stats["std"])

    model.eval()
    return model, ckpt


def compute_metrics(pred: np.ndarray, true: np.ndarray, indices: List[int]) -> Tuple[float, float]:
    if not indices:
        return float("nan"), float("nan")

    p = pred[:, indices]
    t = true[:, indices]

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return mae, rmse


def collect_recursive_blocks(
    model: FlowPredictor,
    series_norm: np.ndarray,
    seq_len: int,
    steps: int,
    device: torch.device,
):
    """
    Non-overlapping recursive blocks.
    Each block starts with real history, predicts `steps`, then resets to real data.
    """
    total_time, num_nodes = series_norm.shape
    if total_time < seq_len + steps:
        return np.array([]), np.array([])

    pred_blocks = []
    true_blocks = []

    # Block starts at target time index s (first predicted point).
    # History is [s-seq_len, s), truth future is [s, s+steps).
    for s in range(seq_len, total_time - steps + 1, steps):
        hist = series_norm[s - seq_len : s]
        future_true = series_norm[s : s + steps]
        hours = np.array([(s + i) % 24 for i in range(steps)], dtype=np.float32)

        x_hist = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)
        y_true = torch.tensor(future_true, dtype=torch.float32, device=device).unsqueeze(0)
        h = torch.tensor(hours, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Force exogenous truth at every step; endogenous remains recursive predictions.
            y_pred = model(
                x_hist,
                h,
                exogenous_truth=y_true,
                teacher_forcing_ratio=1.0,
            )

        pred_blocks.append(y_pred.squeeze(0).cpu().numpy())
        true_blocks.append(future_true)

    if not pred_blocks:
        return np.array([]), np.array([])

    return np.stack(pred_blocks, axis=0), np.stack(true_blocks, axis=0)


def evaluate_recursive_endogenous(steps: int):
    device = config.get_device()
    print(f"Using device: {device}")

    model, ckpt = load_model_for_eval(device)
    topo_order = ckpt["topo_order"]
    exogenous_nodes = ckpt["exogenous_nodes"]

    loader = FlowDataLoader(config.DATA_DIR)
    test_df = loader.load_flow_data("test.csv")
    test_array = align_flow_to_nodes(test_df, topo_order, strict=True)

    norm_stats = ckpt["norm_stats"]
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    test_norm = (test_array - mean) / std

    pred_norm, true_norm = collect_recursive_blocks(
        model=model,
        series_norm=test_norm,
        seq_len=config.SEQ_LEN,
        steps=steps,
        device=device,
    )

    if pred_norm.size == 0:
        raise ValueError(
            f"No recursive blocks generated: rows={len(test_array)}, seq_len={config.SEQ_LEN}, steps={steps}."
        )

    # [blocks, steps, nodes] -> physical space
    pred = pred_norm * std[None, None, :] + mean[None, None, :]
    true = true_norm * std[None, None, :] + mean[None, None, :]

    exo_set = set(exogenous_nodes)
    exo_indices = [i for i, n in enumerate(topo_order) if n in exo_set]
    endo_indices = [i for i, n in enumerate(topo_order) if n not in exo_set]

    if not endo_indices:
        raise ValueError("No endogenous nodes found in checkpoint.")

    metrics = []
    for h in range(steps):
        pred_h = pred[:, h, :]
        true_h = true[:, h, :]

        mae_endo, rmse_endo = compute_metrics(pred_h, true_h, endo_indices)
        mae_exo, rmse_exo = compute_metrics(pred_h, true_h, exo_indices)
        mae_all, rmse_all = compute_metrics(pred_h, true_h, list(range(len(topo_order))))

        metrics.append(
            {
                "step": h + 1,
                "num_blocks": int(pred.shape[0]),
                "mae_endogenous": mae_endo,
                "rmse_endogenous": rmse_endo,
                "mae_exogenous": mae_exo,
                "rmse_exogenous": rmse_exo,
                "mae_all": mae_all,
                "rmse_all": rmse_all,
            }
        )

    return metrics


def save_metrics(metrics: List[Dict[str, float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "recursive_endo_metrics.csv")
    png_path = os.path.join(output_dir, "recursive_endo_metrics.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    steps = [m["step"] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, [m["mae_endogenous"] for m in metrics], marker="o", label="MAE Endogenous")
    plt.plot(steps, [m["rmse_endogenous"] for m in metrics], marker="o", label="RMSE Endogenous")
    plt.plot(steps, [m["mae_exogenous"] for m in metrics], linestyle="--", alpha=0.7, label="MAE Exogenous")

    plt.xlabel("Recursive Step Within Block")
    plt.ylabel("Error")
    plt.title("Recursive Multi-step Evaluation (Exogenous Ground Truth Forced)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, png_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate endogenous recursive multi-step forecasting with exogenous ground-truth forcing."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Recursive steps per block (model predicts this many steps before reset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="flowEvaluation/results",
        help="Directory to save metrics CSV and figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    metrics = evaluate_recursive_endogenous(steps=args.steps)
    csv_path, png_path = save_metrics(metrics, args.output_dir)

    print("=" * 74)
    print(
        "Recursive endogenous evaluation done "
        f"(steps per block={args.steps}, exogenous forced by ground truth)."
    )
    print(f"CSV: {csv_path}")
    print(f"Figure: {png_path}")
    print("Per-step endogenous metrics:")
    for row in metrics:
        print(
            f"step={row['step']:>2d} | "
            f"MAE_endo={row['mae_endogenous']:.6f} | "
            f"RMSE_endo={row['rmse_endogenous']:.6f} | "
            f"blocks={row['num_blocks']}"
        )
    print("=" * 74)


if __name__ == "__main__":
    main()
