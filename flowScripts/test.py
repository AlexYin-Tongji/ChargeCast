"""Test script for two-stage traffic flow model."""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.alignment import align_flow_to_nodes, crop_adjacency
from src.data.flow_data import FlowDataLoader, create_sequences
from src.models.flow_predictor import FlowPredictor
from src.models.topology import TopologyProcessor


def get_loss_fn() -> nn.Module:
    loss_type = config.TRAIN.get("loss", "Huber")
    if loss_type == "Huber":
        return nn.HuberLoss()
    if loss_type == "MAE":
        return nn.L1Loss()
    return nn.MSELoss()


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
    endogenous_nodes = checkpoint.get("endogenous_nodes")
    upstream_dict = checkpoint.get("upstream_dict", {})
    norm_stats = checkpoint.get("norm_stats")

    if norm_stats is None:
        raise ValueError("Checkpoint missing norm_stats.")

    if endogenous_nodes is None:
        exo_set = set(exogenous_nodes)
        endogenous_nodes = [n for n in topo_order if n not in exo_set]

    alignment_stats = checkpoint.get("alignment_stats")
    if alignment_stats is not None:
        print("Alignment stats from checkpoint:", alignment_stats)

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
    print(f"Nodes: total={len(topo_order)}, exogenous={len(exogenous_nodes)}, endogenous={len(endogenous_nodes)}")

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
        torch.tensor(hours_test[:, None], dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.ENDOGENOUS_TRAIN["batch_size"], shuffle=False)

    criterion = get_loss_fn()
    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()

    endo_indices = [topo_order.index(n) for n in endogenous_nodes]
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32, device=device)

    total_denorm_loss = 0.0
    total_norm_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, yb, hb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            hb = hb.to(device)

            pred = model(xb, hb)

            pred_norm = pred[:, :, endo_indices]
            y_norm = yb[:, :, endo_indices]

            pred_denorm = pred * std + mean
            y_denorm = yb * std + mean
            pred_denorm = pred_denorm[:, :, endo_indices]
            y_denorm = y_denorm[:, :, endo_indices]

            batch_size = xb.shape[0]
            total_samples += batch_size

            total_norm_loss += criterion(pred_norm, y_norm).item() * batch_size
            total_denorm_loss += criterion(pred_denorm, y_denorm).item() * batch_size
            total_mae += mae_fn(pred_denorm, y_denorm).item() * batch_size
            total_mse += mse_fn(pred_denorm, y_denorm).item() * batch_size

    avg_norm_loss = total_norm_loss / max(1, total_samples)
    avg_denorm_loss = total_denorm_loss / max(1, total_samples)
    avg_mae = total_mae / max(1, total_samples)
    avg_rmse = float(np.sqrt(total_mse / max(1, total_samples)))

    print("=" * 60)
    print(f"Test Loss ({config.TRAIN.get('loss', 'Huber')}, normalized, endogenous): {avg_norm_loss:.6f}")
    print(f"Test Loss ({config.TRAIN.get('loss', 'Huber')}, denorm, endogenous): {avg_denorm_loss:.6f}")
    print(f"MAE (denorm, endogenous): {avg_mae:.6f}")
    print(f"RMSE (denorm, endogenous): {avg_rmse:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
