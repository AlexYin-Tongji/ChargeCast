"""Train per-station charge models on training data only.

Models:
1) Ratio model: Beta distribution with learnable alpha/beta from Fourier features.
2) Power model: LogNormal distribution with learnable mu/sigma from Fourier features.

Each station/node is trained independently, without train/val/test split.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    hour_period: float = 24.0
    week_period: float = 7.0
    frequencies: Tuple[int, ...] = (1, 2)
    ratio_epochs: int = 400
    power_epochs: int = 400
    ratio_lr: float = 1e-2
    power_lr: float = 1e-2
    min_samples_per_station: int = 20
    min_beta_target: float = 1e-4
    max_beta_target: float = 1.0 - 1e-4
    min_power_target: float = 1e-6
    min_concentration: float = 1e-4
    min_sigma: float = 1e-4
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_node_id(node_id: str) -> str:
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in str(node_id)])


def fourier_features(hour_code: np.ndarray, week_code: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    hour = np.asarray(hour_code, dtype=np.float32)
    week = np.asarray(week_code, dtype=np.float32)

    feats = [np.ones_like(hour, dtype=np.float32)]
    for k in cfg.frequencies:
        feats.append(np.sin(2.0 * np.pi * k * hour / cfg.hour_period).astype(np.float32))
        feats.append(np.cos(2.0 * np.pi * k * hour / cfg.hour_period).astype(np.float32))
        feats.append(np.sin(2.0 * np.pi * k * week / cfg.week_period).astype(np.float32))
        feats.append(np.cos(2.0 * np.pi * k * week / cfg.week_period).astype(np.float32))

    return np.stack(feats, axis=1)


class BetaFourierModel(nn.Module):
    def __init__(self, in_dim: int, min_concentration: float):
        super().__init__()
        self.alpha_head = nn.Linear(in_dim, 1)
        self.beta_head = nn.Linear(in_dim, 1)
        self.min_concentration = min_concentration

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.alpha_head(x)).squeeze(-1) + self.min_concentration
        beta = F.softplus(self.beta_head(x)).squeeze(-1) + self.min_concentration
        return alpha, beta


class LogNormalFourierModel(nn.Module):
    def __init__(self, in_dim: int, min_sigma: float):
        super().__init__()
        self.mu_head = nn.Linear(in_dim, 1)
        self.sigma_head = nn.Linear(in_dim, 1)
        self.min_sigma = min_sigma

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_head(x).squeeze(-1)
        sigma = F.softplus(self.sigma_head(x)).squeeze(-1) + self.min_sigma
        return mu, sigma


def _ratio_targets(y: np.ndarray, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return torch.clamp(y_t, min=cfg.min_beta_target, max=cfg.max_beta_target)


def _power_targets(y: np.ndarray, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return torch.clamp(y_t, min=cfg.min_power_target)


def _eval_ratio(model: BetaFourierModel, X: np.ndarray, y: np.ndarray, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = _ratio_targets(y, cfg, device)
    with torch.no_grad():
        alpha, beta = model(x)
        dist = torch.distributions.Beta(alpha, beta)
        nll = float((-dist.log_prob(y_t).mean()).item())
        mean_pred = alpha / (alpha + beta)
        mae = float(torch.mean(torch.abs(mean_pred - y_t)).item())
        rmse = float(torch.sqrt(torch.mean((mean_pred - y_t) ** 2)).item())
    return {"nll": nll, "mae": mae, "rmse": rmse, "samples": int(len(y))}


def _eval_power(model: LogNormalFourierModel, X: np.ndarray, y: np.ndarray, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = _power_targets(y, cfg, device)
    with torch.no_grad():
        mu, sigma = model(x)
        dist = torch.distributions.LogNormal(mu, sigma)
        nll = float((-dist.log_prob(y_t).mean()).item())
        mean_pred = torch.exp(mu + 0.5 * sigma * sigma)
        mae = float(torch.mean(torch.abs(mean_pred - y_t)).item())
        rmse = float(torch.sqrt(torch.mean((mean_pred - y_t) ** 2)).item())
    return {"nll": nll, "mae": mae, "rmse": rmse, "samples": int(len(y))}


def train_ratio_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[BetaFourierModel, Dict[str, float]]:
    x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = _ratio_targets(y_train, cfg, device)

    model = BetaFourierModel(in_dim=x_train.shape[1], min_concentration=cfg.min_concentration).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ratio_lr)

    for _ in range(cfg.ratio_epochs):
        model.train()
        optimizer.zero_grad()
        alpha, beta = model(x_train)
        dist = torch.distributions.Beta(alpha, beta)
        nll = -dist.log_prob(y_train_t).mean()
        nll.backward()
        optimizer.step()

    model.eval()
    train_eval = _eval_ratio(model, X_train, y_train, cfg, device)
    train_eval["epochs_ran"] = int(cfg.ratio_epochs)
    return model, train_eval


def train_power_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[LogNormalFourierModel, Dict[str, float]]:
    x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = _power_targets(y_train, cfg, device)

    model = LogNormalFourierModel(in_dim=x_train.shape[1], min_sigma=cfg.min_sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.power_lr)

    for _ in range(cfg.power_epochs):
        model.train()
        optimizer.zero_grad()
        mu, sigma = model(x_train)
        dist = torch.distributions.LogNormal(mu, sigma)
        nll = -dist.log_prob(y_train_t).mean()
        nll.backward()
        optimizer.step()

    model.eval()
    train_eval = _eval_power(model, X_train, y_train, cfg, device)
    train_eval["epochs_ran"] = int(cfg.power_epochs)
    return model, train_eval


def train_per_station(
    train_df: pd.DataFrame,
    output_dir: str,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, object]:
    ratio_dir = os.path.join(output_dir, "ratio")
    power_dir = os.path.join(output_dir, "power")
    ensure_dir(ratio_dir)
    ensure_dir(power_dir)

    required_cols = {"hour_code", "week_code", "station_id", "node_id", "charge_flow_ratio", "charge_power"}
    missing = [c for c in required_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"train file missing required columns: {missing}")

    df = train_df.copy()
    df["node_id"] = df["node_id"].astype(str)
    node_ids = sorted(df["node_id"].dropna().unique().tolist())

    summary_rows = []
    skipped_nodes = []

    for node_id in node_ids:
        node_train = df[df["node_id"] == node_id].copy().reset_index(drop=True)

        for col in ["hour_code", "week_code", "charge_flow_ratio", "charge_power"]:
            node_train[col] = pd.to_numeric(node_train[col], errors="coerce")
        node_train = node_train.dropna(subset=["hour_code", "week_code", "charge_flow_ratio", "charge_power"]).reset_index(drop=True)

        if len(node_train) < max(cfg.min_samples_per_station, 2):
            skipped_nodes.append(
                {
                    "node_id": node_id,
                    "reason": f"insufficient_valid_samples<{cfg.min_samples_per_station}",
                    "samples": int(len(node_train)),
                }
            )
            continue

        X_train = fourier_features(node_train["hour_code"].values, node_train["week_code"].values, cfg)
        y_ratio_train = node_train["charge_flow_ratio"].values.astype(np.float32)
        y_power_train = node_train["charge_power"].values.astype(np.float32)

        ratio_model, ratio_metrics = train_ratio_model(X_train, y_ratio_train, cfg, device)
        power_model, power_metrics = train_power_model(X_train, y_power_train, cfg, device)

        safe_node = sanitize_node_id(node_id)
        ratio_path = os.path.join(ratio_dir, f"{safe_node}.pt")
        power_path = os.path.join(power_dir, f"{safe_node}.pt")

        ratio_ckpt = {
            "model_type": "beta_fourier_ratio",
            "node_id": node_id,
            "features": {
                "hour_period": cfg.hour_period,
                "week_period": cfg.week_period,
                "frequencies": list(cfg.frequencies),
                "feature_dim": int(X_train.shape[1]),
            },
            "state_dict": ratio_model.state_dict(),
            "train_metrics": ratio_metrics,
            "config": asdict(cfg),
        }

        power_ckpt = {
            "model_type": "lognormal_fourier_power",
            "node_id": node_id,
            "features": {
                "hour_period": cfg.hour_period,
                "week_period": cfg.week_period,
                "frequencies": list(cfg.frequencies),
                "feature_dim": int(X_train.shape[1]),
            },
            "state_dict": power_model.state_dict(),
            "train_metrics": power_metrics,
            "config": asdict(cfg),
        }

        torch.save(ratio_ckpt, ratio_path)
        torch.save(power_ckpt, power_path)

        summary_rows.append(
            {
                "node_id": node_id,
                "train_samples": int(ratio_metrics["samples"]),
                "ratio_train_nll": ratio_metrics["nll"],
                "ratio_train_mae": ratio_metrics["mae"],
                "ratio_train_rmse": ratio_metrics["rmse"],
                "ratio_epochs": ratio_metrics["epochs_ran"],
                "power_train_nll": power_metrics["nll"],
                "power_train_mae": power_metrics["mae"],
                "power_train_rmse": power_metrics["rmse"],
                "power_epochs": power_metrics["epochs_ran"],
                "ratio_model_path": ratio_path,
                "power_model_path": power_path,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "training_summary.csv")
    summary_json = os.path.join(output_dir, "training_report.json")

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    report = {
        "num_nodes_in_train": int(len(node_ids)),
        "num_nodes_trained": int(len(summary_rows)),
        "num_nodes_skipped": int(len(skipped_nodes)),
        "skipped_nodes": skipped_nodes,
        "summary_csv": summary_csv,
        "config": asdict(cfg),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report["summary_json"] = summary_json
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-station charge ratio/power models.")
    parser.add_argument("--train-path", default="data/charge/train.csv")
    parser.add_argument("--output-dir", default="chargePrediction/models")
    parser.add_argument("--ratio-epochs", type=int, default=400)
    parser.add_argument("--power-epochs", type=int, default=400)
    parser.add_argument("--ratio-lr", type=float, default=1e-2)
    parser.add_argument("--power-lr", type=float, default=1e-2)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="cpu/cuda; empty means auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        ratio_epochs=args.ratio_epochs,
        power_epochs=args.power_epochs,
        ratio_lr=args.ratio_lr,
        power_lr=args.power_lr,
        min_samples_per_station=args.min_samples,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_path)

    ensure_dir(args.output_dir)
    report = train_per_station(train_df, args.output_dir, cfg, device)

    print("=" * 72)
    print("Per-station charge models trained")
    print("=" * 72)
    print(f"device: {device}")
    print(f"num_nodes_in_train: {report['num_nodes_in_train']}")
    print(f"num_nodes_trained: {report['num_nodes_trained']}")
    print(f"num_nodes_skipped: {report['num_nodes_skipped']}")
    print(f"summary_csv: {report['summary_csv']}")
    print(f"summary_json: {report['summary_json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
