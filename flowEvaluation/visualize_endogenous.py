"""Endogenous recursive multi-step visualization on test set.

Protocol per block:
1. Use real history of length SEQ_LEN.
2. Recursively predict `steps` points.
3. Exogenous nodes are forced with ground-truth at each recursive step.
4. Endogenous nodes use model recursive predictions.
5. After one block, reset with real data and continue next block.
"""

from __future__ import annotations

import argparse
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


def load_model_for_vis(device: torch.device) -> Tuple[FlowPredictor, Dict]:
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


def prepare_test_series(loader: FlowDataLoader, node_order: List[str], norm_stats: Dict[str, np.ndarray]):
    test_df = loader.load_flow_data("test.csv")
    test_array = align_flow_to_nodes(test_df, node_order, strict=True)

    mean = norm_stats["mean"]
    std = norm_stats["std"]
    test_norm = (test_array - mean) / std

    return test_norm, test_array


def collect_recursive_blocks(
    model: FlowPredictor,
    series_norm: np.ndarray,
    seq_len: int,
    steps: int,
    device: torch.device,
):
    total_time = len(series_norm)
    if total_time < seq_len + steps:
        return np.array([]), np.array([])

    pred_blocks = []
    true_blocks = []

    for s in range(seq_len, total_time - steps + 1, steps):
        hist = series_norm[s - seq_len : s]
        future_true = series_norm[s : s + steps]
        hours = np.array([(s + i) % 24 for i in range(steps)], dtype=np.float32)

        x_hist = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)
        y_true = torch.tensor(future_true, dtype=torch.float32, device=device).unsqueeze(0)
        h = torch.tensor(hours, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize endogenous recursive multi-step predictions.")
    parser.add_argument("--steps", type=int, default=6, help="Recursive steps per block.")
    parser.add_argument("--max-nodes", type=int, default=16, help="Maximum endogenous nodes to plot.")
    parser.add_argument("--max-points", type=int, default=300, help="Max flattened points per subplot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for node selection.")
    parser.add_argument(
        "--output",
        type=str,
        default="flowEvaluation/results/endogenous_recursive_visualization.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    device = config.get_device()
    print(f"Using device: {device}")

    model, ckpt = load_model_for_vis(device)
    topo_order = ckpt["topo_order"]
    exo_set = set(ckpt["exogenous_nodes"])
    norm_stats = ckpt["norm_stats"]

    loader = FlowDataLoader(config.DATA_DIR)
    test_norm, _ = prepare_test_series(loader, topo_order, norm_stats)

    pred_norm, true_norm = collect_recursive_blocks(
        model=model,
        series_norm=test_norm,
        seq_len=config.SEQ_LEN,
        steps=args.steps,
        device=device,
    )

    if pred_norm.size == 0:
        raise ValueError(
            f"No recursive blocks generated: rows={len(test_norm)}, seq_len={config.SEQ_LEN}, steps={args.steps}."
        )

    mean = norm_stats["mean"]
    std = norm_stats["std"]
    y_true = true_norm * std[None, None, :] + mean[None, None, :]
    y_pred = pred_norm * std[None, None, :] + mean[None, None, :]

    endo_indices = [i for i, n in enumerate(topo_order) if n not in exo_set]
    if not endo_indices:
        raise ValueError("No endogenous nodes found in checkpoint.")

    np.random.seed(args.seed)
    selected = np.random.choice(
        endo_indices,
        size=min(args.max_nodes, len(endo_indices)),
        replace=False,
    ).tolist()

    n = len(selected)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    num_blocks = y_true.shape[0]

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        node_idx = selected[i]
        node_id = topo_order[node_idx]

        true_values = y_true[:, :, node_idx].reshape(-1)
        pred_values = y_pred[:, :, node_idx].reshape(-1)

        limit = min(args.max_points, len(true_values))
        t = np.arange(limit)
        true_values = true_values[:limit]
        pred_values = pred_values[:limit]

        ax.plot(t, true_values, label="True", linewidth=1.4, alpha=0.85, color="tab:blue")
        ax.plot(t, pred_values, label="Pred", linewidth=1.2, alpha=0.85, linestyle="--", color="tab:orange")

        # Visual boundary of each recursive block.
        max_block_lines = int(np.ceil(limit / args.steps))
        for b in range(1, max_block_lines):
            x = b * args.steps - 0.5
            if x < limit:
                ax.axvline(x=x, color="gray", linestyle=":", linewidth=0.8, alpha=0.35)

        ax.set_title(f"{node_id} [ENDO]", fontsize=10)
        ax.set_xlabel("Recursive Time Index")
        ax.set_ylabel("Flow")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Endogenous Recursive Multi-step Visualization\n"
        f"steps={args.steps}, blocks={num_blocks}, exogenous=ground-truth forced",
        fontsize=14,
    )
    fig.tight_layout()

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Recursive blocks: {num_blocks}")
    print(f"Selected endogenous nodes: {n}/{len(endo_indices)}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
