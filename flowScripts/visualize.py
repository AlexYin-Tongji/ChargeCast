"""Visualization script for two-stage traffic flow model."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.alignment import align_flow_to_nodes, crop_adjacency
from src.data.flow_data import FlowDataLoader, create_sequences
from src.models.flow_predictor import FlowPredictor
from src.models.topology import TopologyProcessor


def prepare_test_data(loader: FlowDataLoader, node_order: list, norm_stats: dict):
    test_df = loader.load_flow_data("test.csv")
    test_array = align_flow_to_nodes(test_df, node_order, strict=True)

    mean = norm_stats["mean"]
    std = norm_stats["std"]
    normalized = (test_array - mean) / std

    X, Y = create_sequences(normalized, config.SEQ_LEN, config.PRED_LEN)
    if len(X) == 0:
        raise ValueError(
            f"No test sequences generated. Need rows > SEQ_LEN+PRED_LEN-1, got {len(test_df)} rows."
        )

    hours = np.arange(len(X), dtype=np.float32) % 24
    return X, Y, hours


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    model_path = os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    topo_order = checkpoint["topo_order"]
    canonical_nodes = checkpoint.get("canonical_nodes", topo_order)
    exogenous_nodes = checkpoint["exogenous_nodes"]
    upstream_dict = checkpoint.get("upstream_dict", {})
    norm_stats = checkpoint["norm_stats"]

    if not upstream_dict:
        loader = FlowDataLoader(config.DATA_DIR)
        adj_df = loader.load_adjacency("adj.csv")
        try:
            cropped_adj = crop_adjacency(adj_df, canonical_nodes)
        except KeyError as exc:
            raise ValueError(
                "Checkpoint nodes are incompatible with current adj.csv. "
                "Please retrain with the new two-stage pipeline."
            ) from exc
        topo = TopologyProcessor(cropped_adj)
        upstream_dict = {node: topo.get_upstream_nodes(node) for node in topo_order}

    model = FlowPredictor(
        num_nodes=len(topo_order),
        node_ids=topo_order,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=upstream_dict,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.set_normalization(norm_stats["mean"], norm_stats["std"])
    model.eval()

    loader = FlowDataLoader(config.DATA_DIR)
    X_test, Y_test, hours_test = prepare_test_data(loader, topo_order, norm_stats)

    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    hours_tensor = torch.tensor(hours_test[:, None], dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(X_tensor, hours_tensor).cpu().numpy()

    mean = norm_stats["mean"]
    std = norm_stats["std"]
    y_true = Y_test * std + mean
    y_pred = predictions * std + mean

    exo_set = set(exogenous_nodes)
    exo_indices = [i for i, n in enumerate(topo_order) if n in exo_set]
    endo_indices = [i for i, n in enumerate(topo_order) if n not in exo_set]

    np.random.seed(42)
    selected_exo = (
        np.random.choice(exo_indices, size=min(4, len(exo_indices)), replace=False).tolist()
        if exo_indices
        else []
    )
    remaining = max(0, 16 - len(selected_exo))
    selected_endo = (
        np.random.choice(endo_indices, size=min(remaining, len(endo_indices)), replace=False).tolist()
        if endo_indices
        else []
    )
    selected = selected_exo + selected_endo

    if not selected:
        raise ValueError("No nodes available for visualization after alignment.")

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

    max_points = min(200, y_true.shape[0])
    x_axis = np.arange(max_points)

    for i in range(rows * cols):
        ax = axes[i]
        if i >= len(selected):
            ax.axis("off")
            continue

        node_idx = selected[i]
        node_id = topo_order[node_idx]
        is_exo = node_id in exo_set

        true_values = y_true[:max_points, 0, node_idx]
        pred_values = y_pred[:max_points, 0, node_idx]

        ax.plot(x_axis, true_values, label="True", linewidth=1.4, alpha=0.8)
        ax.plot(
            x_axis,
            pred_values,
            label="Pred",
            linewidth=1.2,
            alpha=0.8,
            linestyle="--",
            color=("red" if is_exo else "orange"),
        )

        role = "EXO" if is_exo else "ENDO"
        ax.set_title(f"{node_id} [{role}]", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flow")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title = (
        "Traffic Flow Prediction (Aligned Canonical Nodes)\n"
        f"total={len(topo_order)}, exogenous={len(exogenous_nodes)}, endogenous={len(endo_indices)}"
    )
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    out_path = os.path.join(config.MODEL_DIR, "visualization.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    main()
