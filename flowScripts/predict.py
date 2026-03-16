"""Inference helpers for two-stage traffic flow model."""

import os
import sys
from typing import Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.flow_data import FlowDataLoader
from src.models.flow_predictor import FlowPredictor
from src.models.topology import TopologyProcessor


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[FlowPredictor, dict]:
    """Load trained stage-2 model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    topo_order = checkpoint["topo_order"]
    exogenous_nodes = checkpoint["exogenous_nodes"]

    upstream_dict = checkpoint.get("upstream_dict")
    if not upstream_dict:
        loader = FlowDataLoader(config.DATA_DIR)
        adj_df = loader.load_adjacency("adj.csv")
        topo = TopologyProcessor(adj_df)
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
    if "norm_stats" in checkpoint and checkpoint["norm_stats"] is not None:
        model.set_normalization(checkpoint["norm_stats"]["mean"], checkpoint["norm_stats"]["std"])
    model.eval()

    return model, checkpoint


def predict(
    model: FlowPredictor,
    history_data: np.ndarray,
    hours: np.ndarray,
    norm_stats: dict,
    device: torch.device,
) -> np.ndarray:
    """Predict flow for one or more future steps."""
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    normalized_data = (history_data - mean) / std

    x_hist = torch.tensor(normalized_data, dtype=torch.float32, device=device).unsqueeze(0)
    hours_tensor = torch.tensor(hours, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        output = model(x_hist, hours_tensor)

    output = output.squeeze(0).cpu().numpy()
    output = output * std + mean

    return output


def predict_next_hour(
    model: FlowPredictor,
    history_data: np.ndarray,
    current_hour: int,
    norm_stats: dict,
    device: torch.device,
) -> np.ndarray:
    next_hour = (current_hour + 1) % 24
    return predict(model, history_data, np.array([next_hour], dtype=np.float32), norm_stats, device)


def predict_multi_steps(
    model: FlowPredictor,
    history_data: np.ndarray,
    start_hour: int,
    pred_len: int,
    norm_stats: dict,
    device: torch.device,
) -> np.ndarray:
    hours = np.array([(start_hour + i) % 24 for i in range(pred_len)], dtype=np.float32)
    return predict(model, history_data, hours, norm_stats, device)


def demo():
    device = config.get_device()
    print(f"Using device: {device}")

    model_path = os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"])
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first: python flowScripts/train.py")
        return

    model, checkpoint = load_model(model_path, device)
    norm_stats = checkpoint["norm_stats"]

    seq_len = config.SEQ_LEN
    num_nodes = len(checkpoint["topo_order"])

    np.random.seed(42)
    history_data = np.random.randn(seq_len, num_nodes) * 10 + 50

    out = predict_multi_steps(
        model,
        history_data,
        start_hour=12,
        pred_len=3,
        norm_stats=norm_stats,
        device=device,
    )

    print("Prediction shape:", out.shape)
    print("Head rows:")
    print(out[:2, : min(5, out.shape[1])])


if __name__ == "__main__":
    demo()
