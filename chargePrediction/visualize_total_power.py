"""Visualize end-to-end total charge power prediction on test split."""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from chargePrediction.test_station_models import run_end_to_end_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize end-to-end total charge power prediction.")
    parser.add_argument("--charge-path", default="data/charge/charge.csv")
    parser.add_argument("--flow-train-path", default="data/flow/train.csv")
    parser.add_argument("--flow-test-path", default="data/flow/test.csv")
    parser.add_argument(
        "--flow-checkpoint-path",
        default=os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"]),
    )
    parser.add_argument("--station-model-dir", default="chargePrediction/models")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--model-pred-batch-size", type=int, default=256)
    parser.add_argument("--use-canonical-filter", action="store_true")
    parser.add_argument("--no-clip-negative-flow-pred", action="store_true")
    parser.add_argument("--max-stations", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="chargePrediction/models/e2e_total_power_visualization.png")
    parser.add_argument("--device", default="", help="cpu/cuda; empty means auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = config.get_device()

    result = run_end_to_end_test(
        charge_path=args.charge_path,
        flow_train_path=args.flow_train_path,
        flow_test_path=args.flow_test_path,
        flow_checkpoint_path=args.flow_checkpoint_path,
        station_model_dir=args.station_model_dir,
        test_ratio=args.test_ratio,
        use_canonical_filter=args.use_canonical_filter,
        clip_negative_flow_pred=not args.no_clip_negative_flow_pred,
        model_pred_batch_size=args.model_pred_batch_size,
        device=device,
    )

    detail_df: pd.DataFrame = result["detail_df"].copy()
    report = result["report"]

    detail_df["_time"] = pd.to_datetime(detail_df["date"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(detail_df["hour_code"], errors="coerce").fillna(0).astype(int),
        unit="h",
    )
    detail_df = detail_df.sort_values(["_time", "node_id"]).reset_index(drop=True)

    station_counts = detail_df.groupby("node_id").size().sort_values(ascending=False)
    all_nodes = station_counts.index.tolist()

    if not all_nodes:
        raise ValueError("No nodes available for visualization.")

    np.random.seed(args.seed)
    selected = all_nodes[: max(1, args.max_stations)]

    n = len(selected)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, node_id in enumerate(selected):
        ax = axes[i]
        node_df = detail_df[detail_df["node_id"] == node_id].copy().sort_values("_time")
        if len(node_df) > args.max_points:
            node_df = node_df.iloc[: args.max_points]

        x = np.arange(len(node_df))
        ax.plot(x, node_df["true_total_power"].values, label="True", linewidth=1.4, alpha=0.85)
        ax.plot(
            x,
            node_df["pred_total_power"].values,
            label="Pred",
            linewidth=1.2,
            linestyle="--",
            color="tab:orange",
            alpha=0.85,
        )
        ax.set_title(f"{node_id} (n={len(node_df)})", fontsize=9)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Total Power")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "End-to-End Total Charge Power Prediction\n"
        f"MAE={report['metrics']['mae_total_power']:.4f}, RMSE={report['metrics']['rmse_total_power']:.4f}, "
        f"test_rows={report['counts']['rows_evaluated']}",
        fontsize=12,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()

    print("=" * 72)
    print("End-to-end total power visualization saved")
    print("=" * 72)
    print(f"device: {device}")
    print(f"nodes_plotted: {len(selected)}")
    print(f"output_path: {args.output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
