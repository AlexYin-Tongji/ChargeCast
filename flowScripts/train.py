"""Two-stage training script.

Stage 1: train one LSTM per exogenous node.
Stage 2: freeze exogenous models, train endogenous propagation/allocation.
"""

import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.alignment import (
    align_flow_to_nodes,
    build_alignment_report,
    build_canonical_nodes,
    crop_adjacency,
)
from src.data.flow_data import FlowDataLoader, create_sequences
from src.models.flow_predictor import FlowPredictor
from src.models.topology import TopologyProcessor


class SingleNodeLSTM(nn.Module):
    """Single exogenous node LSTM for stage-1 training."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def get_loss_fn() -> nn.Module:
    loss_type = config.TRAIN.get("loss", "Huber")
    if loss_type == "Huber":
        return nn.HuberLoss()
    if loss_type == "MAE":
        return nn.L1Loss()
    return nn.MSELoss()


def scheduled_sampling_ratio(epoch: int) -> float:
    cfg = config.SCHEDULED_SAMPLING
    if not cfg.get("enabled", True):
        return 0.0

    start_ratio = float(cfg.get("start_ratio", 1.0))
    end_ratio = float(cfg.get("end_ratio", 0.2))
    start_epoch = int(cfg.get("start_epoch", 0))
    end_epoch = int(cfg.get("end_epoch", 80))

    if epoch <= start_epoch:
        return start_ratio
    if epoch >= end_epoch:
        return end_ratio
    if end_epoch == start_epoch:
        return end_ratio

    alpha = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return start_ratio + alpha * (end_ratio - start_ratio)


def sanitize_node_id(node_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(node_id))


def node_checkpoint_path(save_dir: str, node_id: str) -> str:
    return os.path.join(save_dir, f"{sanitize_node_id(node_id)}.pt")


def build_training_context(data_dir: str):
    loader = FlowDataLoader(data_dir)
    train_df = loader.load_flow_data("train.csv")
    test_df = loader.load_flow_data("test.csv")
    adj_df = loader.load_adjacency("adj.csv")

    canonical_nodes, stats = build_canonical_nodes(
        adj_df,
        train_df,
        test_df,
        enabled=config.NODE_ALIGNMENT.get("enabled", True),
    )

    if len(canonical_nodes) == 0:
        raise ValueError("No canonical nodes found from adj ∩ train ∩ test.")

    adj_canonical = crop_adjacency(adj_df, canonical_nodes)
    topo = TopologyProcessor(adj_canonical)
    topo_order = topo.get_topological_order()

    upstream_dict = {}
    for node in topo_order:
        up_nodes = topo.get_upstream_nodes(node)
        if not isinstance(up_nodes, list):
            up_nodes = list(up_nodes) if hasattr(up_nodes, "__iter__") else []
        upstream_dict[node] = up_nodes

    alignment_report = build_alignment_report(
        stats,
        preview_count=config.NODE_ALIGNMENT.get("dropped_preview_count", 20),
    )

    return {
        "loader": loader,
        "train_df": train_df,
        "test_df": test_df,
        "topo": topo,
        "topo_order": topo_order,
        "adj_canonical": adj_canonical,
        "canonical_nodes": canonical_nodes,
        "upstream_dict": upstream_dict,
        "alignment_report": alignment_report,
    }


def prepare_sequences(train_df, node_order: List[str]):
    all_data = align_flow_to_nodes(train_df, node_order, strict=True)

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8
    normalized_data = (all_data - mean) / std

    X, Y = create_sequences(normalized_data, config.SEQ_LEN, config.PRED_LEN)
    if len(X) == 0:
        raise ValueError(
            f"No training sequences generated. Need rows > SEQ_LEN+PRED_LEN-1, got {len(train_df)} rows."
        )

    train_size = int(len(X) * config.TRAIN_RATIO)
    train_size = max(1, min(train_size, len(X) - 1)) if len(X) > 1 else 1

    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    if len(X_val) == 0:
        X_val, Y_val = X_train.copy(), Y_train.copy()

    hours = np.arange(len(X), dtype=np.float32) % 24
    hours_train, hours_val = hours[: len(X_train)], hours[len(X_train) :]
    if len(hours_val) == 0:
        hours_val = hours_train.copy()

    norm_stats = {"mean": mean, "std": std}
    return X_train, Y_train, X_val, Y_val, hours_train, hours_val, norm_stats


def train_exogenous_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    exogenous_nodes: List[str],
    node_order: List[str],
    device: torch.device,
):
    save_dir = config.EXOGENOUS_TRAIN["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    criterion = get_loss_fn()
    node_to_idx = {n: i for i, n in enumerate(node_order)}

    for node_id in exogenous_nodes:
        idx = node_to_idx[node_id]
        model = SingleNodeLSTM(
            hidden_size=config.LSTM["hidden_size"],
            num_layers=config.LSTM["num_layers"],
            dropout=config.LSTM.get("dropout", 0.0),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.EXOGENOUS_TRAIN["learning_rate"],
            weight_decay=config.EXOGENOUS_TRAIN.get("weight_decay", 0.0),
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train[:, :, idx], dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y_train[:, 0, idx], dtype=torch.float32).unsqueeze(-1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val[:, :, idx], dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y_val[:, 0, idx], dtype=torch.float32).unsqueeze(-1),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.EXOGENOUS_TRAIN["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.EXOGENOUS_TRAIN["batch_size"],
            shuffle=False,
        )

        best_val = float("inf")
        patience = 0

        for epoch in range(config.EXOGENOUS_TRAIN["epochs"]):
            model.train()
            total_train = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_train += loss.item()

            train_loss = total_train / max(1, len(train_loader))

            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    total_val += loss.item()

            val_loss = total_val / max(1, len(val_loader))

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                ckpt_path = node_checkpoint_path(save_dir, node_id)
                torch.save(
                    {
                        "node_id": node_id,
                        "node_index": idx,
                        "best_val_loss": best_val,
                        "lstm_state_dict": model.lstm.state_dict(),
                        "fc_state_dict": model.fc.state_dict(),
                        "hidden_size": config.LSTM["hidden_size"],
                        "num_layers": config.LSTM["num_layers"],
                        "dropout": config.LSTM.get("dropout", 0.0),
                    },
                    ckpt_path,
                )
            else:
                patience += 1

            if patience >= config.EXOGENOUS_TRAIN.get("early_stop_patience", 12):
                break

        print(f"[Stage1] node={node_id} best_val={best_val:.6f}")


def load_exogenous_weights(model: FlowPredictor, exogenous_nodes: List[str], device: torch.device):
    save_dir = config.EXOGENOUS_TRAIN["save_dir"]
    for i, node_id in enumerate(exogenous_nodes):
        path = node_checkpoint_path(save_dir, node_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing exogenous checkpoint: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.exogenous_lstm.lstm_layers[i].load_state_dict(ckpt["lstm_state_dict"])
        model.exogenous_lstm.fc_layers[i].load_state_dict(ckpt["fc_state_dict"])

    for p in model.exogenous_lstm.parameters():
        p.requires_grad = False


def evaluate_endogenous(
    model: FlowPredictor,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    endogenous_indices: List[int],
    norm_stats: Dict[str, np.ndarray],
):
    model.eval()
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32, device=device)

    total = 0.0
    with torch.no_grad():
        for xb, yb, hb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            hb = hb.to(device)

            pred = model(xb, hb, exogenous_truth=None, teacher_forcing_ratio=0.0)
            pred_denorm = pred * std + mean
            y_denorm = yb * std + mean

            loss = criterion(pred_denorm[:, :, endogenous_indices], y_denorm[:, :, endogenous_indices])
            total += loss.item()

    return total / max(1, len(dataloader))


def train_endogenous(
    model: FlowPredictor,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    hours_train: np.ndarray,
    hours_val: np.ndarray,
    norm_stats: Dict[str, np.ndarray],
    canonical_nodes: List[str],
    alignment_report: Dict[str, object],
    upstream_dict: Dict[str, List[str]],
    device: torch.device,
):
    criterion = get_loss_fn()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
        torch.tensor(hours_train[:, None], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
        torch.tensor(hours_val[:, None], dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ENDOGENOUS_TRAIN["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.ENDOGENOUS_TRAIN["batch_size"],
        shuffle=False,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.ENDOGENOUS_TRAIN["learning_rate"],
        weight_decay=config.ENDOGENOUS_TRAIN.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.ENDOGENOUS_TRAIN["lr_factor"],
        patience=config.ENDOGENOUS_TRAIN["lr_patience"],
        min_lr=config.ENDOGENOUS_TRAIN.get("lr_min", 1e-6),
    )

    endogenous_nodes = [n for n in model.node_ids if n not in set(model.exogenous_nodes)]
    endogenous_indices = [model.node_to_idx[n] for n in endogenous_nodes]
    if not endogenous_indices:
        raise ValueError("No endogenous nodes left after alignment; stage-2 cannot train.")

    best_val = float("inf")
    patience = 0

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"])

    for epoch in range(config.ENDOGENOUS_TRAIN["epochs"]):
        model.train()
        ratio = scheduled_sampling_ratio(epoch)

        train_total = 0.0
        for xb, yb, hb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            hb = hb.to(device)

            optimizer.zero_grad()
            pred = model(xb, hb, exogenous_truth=yb, teacher_forcing_ratio=ratio)
            loss = criterion(pred[:, :, endogenous_indices], yb[:, :, endogenous_indices])
            loss.backward()

            grad_clip = config.ENDOGENOUS_TRAIN.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()
            train_total += loss.item()

        train_loss = train_total / max(1, len(train_loader))
        val_loss = evaluate_endogenous(
            model,
            val_loader,
            criterion,
            device,
            endogenous_indices,
            norm_stats,
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "norm_stats": norm_stats,
                    "topo_order": model.topo_order,
                    "canonical_nodes": canonical_nodes,
                    "exogenous_nodes": model.exogenous_nodes,
                    "endogenous_nodes": endogenous_nodes,
                    "upstream_dict": upstream_dict,
                    "alignment_stats": alignment_report,
                    "exogenous_checkpoint_dir": config.EXOGENOUS_TRAIN["save_dir"],
                },
                checkpoint_path,
            )
            marker = " *"
        else:
            patience += 1
            marker = ""

        print(
            f"[Stage2] epoch={epoch+1}/{config.ENDOGENOUS_TRAIN['epochs']} "
            f"ratio={ratio:.3f} train={train_loss:.6f} val={val_loss:.6f}{marker}"
        )

        if patience >= config.ENDOGENOUS_TRAIN.get("early_stop_patience", 20):
            print(f"[Stage2] early stop at epoch {epoch+1}")
            break

    print(f"[Stage2] best_val_loss={best_val:.6f}")


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    context = build_training_context(config.DATA_DIR)
    print("Alignment stats:", context["alignment_report"])

    topo_order = context["topo_order"]
    exogenous_nodes = list(context["topo"].exogenous_nodes)
    endogenous_nodes = list(context["topo"].endogenous_nodes)

    print(
        f"Canonical nodes={len(topo_order)} | "
        f"Exogenous={len(exogenous_nodes)} | Endogenous={len(endogenous_nodes)}"
    )

    X_train, Y_train, X_val, Y_val, hours_train, hours_val, norm_stats = prepare_sequences(
        context["train_df"],
        topo_order,
    )

    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val:   X={X_val.shape}, Y={Y_val.shape}")

    print("\n[Stage1] Training one exogenous model per node...")
    train_exogenous_models(
        X_train,
        Y_train,
        X_val,
        Y_val,
        exogenous_nodes,
        topo_order,
        device,
    )

    print("\n[Stage2] Building full model and loading exogenous checkpoints...")
    model = FlowPredictor(
        num_nodes=len(topo_order),
        node_ids=topo_order,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=context["upstream_dict"],
    ).to(device)
    model.set_normalization(norm_stats["mean"], norm_stats["std"])

    load_exogenous_weights(model, exogenous_nodes, device)

    train_endogenous(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        hours_train,
        hours_val,
        norm_stats,
        context["canonical_nodes"],
        context["alignment_report"],
        context["upstream_dict"],
        device,
    )


if __name__ == "__main__":
    main()
