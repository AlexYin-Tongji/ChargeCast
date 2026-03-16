"""End-to-end test for charge total power prediction.

Pipeline on last 20% time slice:
1) Predict gate flow by trained traffic flow model.
2) Predict charge flow ratio by per-station Beta model.
3) Predict average charge power by per-station LogNormal model.
4) Predict total charge power:
   pred_total_power = pred_gate_flow * pred_ratio * pred_avg_power

Metrics are computed on total charge power only (MAE, RMSE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from flowScripts.predict import load_model
from src.data.alignment import build_canonical_nodes
from src.data.flow_data import FlowDataLoader

from chargePrediction.prepare_charge_dataset import (
    MERGE_KEYS,
    _build_model_pred_table,
    _build_time_col,
    _merge_true_flow_train_then_test,
    _read_flow_raw,
    _to_int_hour,
    _to_str,
)


@dataclass
class EvalDefaults:
    min_concentration: float = 1e-4
    min_sigma: float = 1e-4


def sanitize_node_id(node_id: str) -> str:
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in str(node_id)])


def fourier_features(hour_code: np.ndarray, week_code: np.ndarray, hour_period: float, week_period: float, frequencies: List[int]) -> np.ndarray:
    hour = np.asarray(hour_code, dtype=np.float32)
    week = np.asarray(week_code, dtype=np.float32)
    feats = [np.ones_like(hour, dtype=np.float32)]
    for k in frequencies:
        feats.append(np.sin(2.0 * np.pi * k * hour / hour_period).astype(np.float32))
        feats.append(np.cos(2.0 * np.pi * k * hour / hour_period).astype(np.float32))
        feats.append(np.sin(2.0 * np.pi * k * week / week_period).astype(np.float32))
        feats.append(np.cos(2.0 * np.pi * k * week / week_period).astype(np.float32))
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


def _predict_station_components(
    node_df: pd.DataFrame,
    ratio_ckpt: Dict[str, object],
    power_ckpt: Dict[str, object],
    device: torch.device,
    defaults: EvalDefaults,
) -> pd.DataFrame:
    ratio_feat = ratio_ckpt.get("features", {})
    power_feat = power_ckpt.get("features", {})
    ratio_cfg = ratio_ckpt.get("config", {})
    power_cfg = power_ckpt.get("config", {})

    hour_period = float(ratio_feat.get("hour_period", 24.0))
    week_period = float(ratio_feat.get("week_period", 7.0))
    frequencies = [int(x) for x in ratio_feat.get("frequencies", [1, 2])]
    ratio_dim = int(ratio_feat.get("feature_dim", 1 + 4 * len(frequencies)))
    power_dim = int(power_feat.get("feature_dim", ratio_dim))

    X = fourier_features(
        node_df["hour_code"].values,
        node_df["week_code"].values,
        hour_period=hour_period,
        week_period=week_period,
        frequencies=frequencies,
    )
    x = torch.tensor(X, dtype=torch.float32, device=device)

    ratio_model = BetaFourierModel(
        in_dim=ratio_dim,
        min_concentration=float(ratio_cfg.get("min_concentration", defaults.min_concentration)),
    ).to(device)
    ratio_model.load_state_dict(ratio_ckpt["state_dict"])
    ratio_model.eval()

    power_model = LogNormalFourierModel(
        in_dim=power_dim,
        min_sigma=float(power_cfg.get("min_sigma", defaults.min_sigma)),
    ).to(device)
    power_model.load_state_dict(power_ckpt["state_dict"])
    power_model.eval()

    with torch.no_grad():
        alpha, beta = ratio_model(x)
        pred_ratio = (alpha / (alpha + beta)).cpu().numpy()

        mu, sigma = power_model(x)
        pred_avg_power = torch.exp(mu + 0.5 * sigma * sigma).cpu().numpy()

    out = node_df.copy()
    out["pred_ratio"] = pred_ratio
    out["pred_avg_power"] = pred_avg_power
    out["pred_ratio"] = out["pred_ratio"].clip(lower=0.0, upper=1.0)
    out["pred_avg_power"] = out["pred_avg_power"].clip(lower=0.0)
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_end_to_end_test(
    charge_path: str,
    flow_train_path: str,
    flow_test_path: str,
    flow_checkpoint_path: str,
    station_model_dir: str,
    test_ratio: float,
    use_canonical_filter: bool,
    clip_negative_flow_pred: bool,
    model_pred_batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    defaults = EvalDefaults()

    charge_df = pd.read_csv(charge_path)
    required_charge_cols = {"node_id", "date", "hour_code", "week_code", "nev_flow", "power"}
    missing_charge = [c for c in required_charge_cols if c not in charge_df.columns]
    if missing_charge:
        raise ValueError(f"charge csv missing required columns: {missing_charge}")

    if "station_id" not in charge_df.columns:
        charge_df["station_id"] = charge_df["node_id"]

    charge_df = _to_str(charge_df, ["node_id", "station_id", "date"])
    charge_df = _to_int_hour(charge_df, "hour_code")
    charge_df["week_code"] = pd.to_numeric(charge_df["week_code"], errors="coerce")
    charge_df["nev_flow"] = pd.to_numeric(charge_df["nev_flow"], errors="coerce")
    charge_df["power"] = pd.to_numeric(charge_df["power"], errors="coerce")
    charge_df = charge_df.dropna(subset=["node_id", "date", "hour_code", "week_code", "nev_flow", "power"]).reset_index(drop=True)

    loader = FlowDataLoader(config.DATA_DIR)
    flow_train_pivot = loader.load_flow_data("train.csv")
    flow_test_pivot = loader.load_flow_data("test.csv")
    adj_df = loader.load_adjacency("adj.csv")
    canonical_nodes, canonical_stats = build_canonical_nodes(
        adj_df,
        flow_train_pivot,
        flow_test_pivot,
        enabled=config.NODE_ALIGNMENT.get("enabled", True),
    )
    canonical_set = set(canonical_nodes)

    rows_before_filter = len(charge_df)
    if use_canonical_filter:
        charge_df = charge_df[charge_df["node_id"].isin(canonical_set)].copy()
    rows_after_filter = len(charge_df)

    flow_train_raw = _read_flow_raw(flow_train_path)
    flow_test_raw = _read_flow_raw(flow_test_path)

    base_df = _merge_true_flow_train_then_test(charge_df, flow_train_raw, flow_test_raw)

    flow_model, flow_ckpt = load_model(flow_checkpoint_path, device)
    pred_flow_table = _build_model_pred_table(
        model=flow_model,
        checkpoint=flow_ckpt,
        flow_train=flow_train_raw,
        flow_test=flow_test_raw,
        charge_nodes=base_df["node_id"].dropna().astype(str).unique().tolist(),
        batch_size=model_pred_batch_size,
        clip_negative_pred=clip_negative_flow_pred,
    )
    base_df = base_df.merge(pred_flow_table, on=MERGE_KEYS, how="left")

    base_df = base_df.rename(columns={"nev_flow": "charge_nev_flow", "power": "charge_power_total"})
    base_df["charge_nev_flow"] = pd.to_numeric(base_df["charge_nev_flow"], errors="coerce")
    base_df["charge_power_total"] = pd.to_numeric(base_df["charge_power_total"], errors="coerce")
    base_df["gate_nev_flow_pred"] = pd.to_numeric(base_df["gate_nev_flow_pred"], errors="coerce")

    base_df["_time"] = _build_time_col(base_df)
    base_df = base_df.sort_values(["_time", "node_id"]).reset_index(drop=True)

    n_total = len(base_df)
    if n_total == 0:
        raise ValueError("No rows available after preprocessing.")
    n_test = int(round(n_total * test_ratio))
    n_test = max(1, min(n_test, n_total))
    n_train = n_total - n_test
    test_df = base_df.iloc[n_train:].copy().reset_index(drop=True)

    rows_test_missing_flow_pred = int(test_df["gate_nev_flow_pred"].isna().sum())
    test_df = test_df.dropna(subset=["hour_code", "week_code", "charge_power_total", "gate_nev_flow_pred"]).copy()
    test_df["gate_nev_flow_pred"] = test_df["gate_nev_flow_pred"].clip(lower=0.0)

    ratio_dir = os.path.join(station_model_dir, "ratio")
    power_dir = os.path.join(station_model_dir, "power")
    if not os.path.isdir(ratio_dir) or not os.path.isdir(power_dir):
        raise FileNotFoundError(
            f"Station model directory invalid. ratio_dir={ratio_dir}, power_dir={power_dir}"
        )

    result_rows = []
    skipped_nodes = []

    for node_id, node_df in test_df.groupby("node_id", sort=True):
        node_df = node_df.copy().reset_index(drop=True)
        safe_node = sanitize_node_id(node_id)
        ratio_path = os.path.join(ratio_dir, f"{safe_node}.pt")
        power_path = os.path.join(power_dir, f"{safe_node}.pt")

        if not os.path.exists(ratio_path) or not os.path.exists(power_path):
            skipped_nodes.append(
                {
                    "node_id": node_id,
                    "reason": "missing_checkpoint",
                    "missing_ratio": not os.path.exists(ratio_path),
                    "missing_power": not os.path.exists(power_path),
                    "rows": int(len(node_df)),
                }
            )
            continue

        ratio_ckpt = torch.load(ratio_path, map_location=device, weights_only=False)
        power_ckpt = torch.load(power_path, map_location=device, weights_only=False)

        pred_df = _predict_station_components(node_df, ratio_ckpt, power_ckpt, device, defaults)
        pred_df["pred_gate_flow"] = pred_df["gate_nev_flow_pred"].clip(lower=0.0)
        pred_df["pred_charge_flow"] = pred_df["pred_ratio"] * pred_df["pred_gate_flow"]
        pred_df["pred_total_power"] = pred_df["pred_charge_flow"] * pred_df["pred_avg_power"]
        pred_df["pred_total_power"] = pred_df["pred_total_power"].clip(lower=0.0)
        pred_df["true_total_power"] = pred_df["charge_power_total"].clip(lower=0.0)
        result_rows.append(pred_df)

    if not result_rows:
        raise ValueError("No test rows could be evaluated. Check station checkpoints and data alignment.")

    detail_df = pd.concat(result_rows, axis=0, ignore_index=True)
    detail_df = detail_df.dropna(subset=["pred_total_power", "true_total_power"]).reset_index(drop=True)

    y_true = detail_df["true_total_power"].to_numpy(dtype=np.float64)
    y_pred = detail_df["pred_total_power"].to_numpy(dtype=np.float64)

    mae = float(np.mean(np.abs(y_true - y_pred))) if len(detail_df) else float("nan")
    rmse = _rmse(y_true, y_pred)

    node_metrics = []
    for node_id, node_df in detail_df.groupby("node_id", sort=True):
        yt = node_df["true_total_power"].to_numpy(dtype=np.float64)
        yp = node_df["pred_total_power"].to_numpy(dtype=np.float64)
        node_metrics.append(
            {
                "node_id": node_id,
                "samples": int(len(node_df)),
                "mae_total_power": float(np.mean(np.abs(yt - yp))),
                "rmse_total_power": _rmse(yt, yp),
            }
        )
    node_metrics_df = pd.DataFrame(node_metrics)

    report = {
        "paths": {
            "charge_path": charge_path,
            "flow_train_path": flow_train_path,
            "flow_test_path": flow_test_path,
            "flow_checkpoint_path": flow_checkpoint_path,
            "station_model_dir": station_model_dir,
        },
        "settings": {
            "test_ratio": test_ratio,
            "use_canonical_filter": use_canonical_filter,
            "clip_negative_flow_pred": clip_negative_flow_pred,
            "model_pred_batch_size": model_pred_batch_size,
        },
        "counts": {
            "rows_before_filter": int(rows_before_filter),
            "rows_after_filter": int(rows_after_filter),
            "rows_total": int(n_total),
            "rows_test_split": int(len(test_df)),
            "rows_evaluated": int(len(detail_df)),
            "nodes_evaluated": int(detail_df["node_id"].nunique()),
            "nodes_skipped": int(len(skipped_nodes)),
            "canonical_nodes": int(len(canonical_nodes)),
            "flow_pred_missing_in_test": rows_test_missing_flow_pred,
        },
        "metrics": {
            "mae_total_power": mae,
            "rmse_total_power": rmse,
        },
        "skipped_nodes": skipped_nodes,
        "flow_canonical_stats": {
            "adj_nodes": canonical_stats.get("adj_nodes", 0),
            "train_nodes": canonical_stats.get("train_nodes", 0),
            "test_nodes": canonical_stats.get("test_nodes", 0),
            "canonical_nodes": canonical_stats.get("canonical_nodes", 0),
        },
    }

    return {
        "detail_df": detail_df,
        "node_metrics_df": node_metrics_df,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end test on total charge power (last 20%).")
    parser.add_argument("--charge-path", default="data/charge/charge.csv")
    parser.add_argument("--flow-train-path", default="data/flow/train.csv")
    parser.add_argument("--flow-test-path", default="data/flow/test.csv")
    parser.add_argument(
        "--flow-checkpoint-path",
        default=os.path.join(config.MODEL_DIR, config.ENDOGENOUS_TRAIN["checkpoint_name"]),
    )
    parser.add_argument("--station-model-dir", default="chargePrediction/models")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--output-csv", default="chargePrediction/models/e2e_test_detail.csv")
    parser.add_argument("--output-node-csv", default="chargePrediction/models/e2e_test_node_metrics.csv")
    parser.add_argument("--output-json", default="chargePrediction/models/e2e_test_report.json")
    parser.add_argument("--model-pred-batch-size", type=int, default=256)
    parser.add_argument("--use-canonical-filter", action="store_true")
    parser.add_argument("--no-clip-negative-flow-pred", action="store_true")
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

    detail_df: pd.DataFrame = result["detail_df"]
    node_metrics_df: pd.DataFrame = result["node_metrics_df"]
    report: Dict[str, object] = result["report"]

    for p in [args.output_csv, args.output_node_csv, args.output_json]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    detail_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    node_metrics_df.to_csv(args.output_node_csv, index=False, encoding="utf-8-sig")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("End-to-end charge total power test finished")
    print("=" * 72)
    print(f"device: {device}")
    print(f"rows_test_split: {report['counts']['rows_test_split']}")
    print(f"rows_evaluated:  {report['counts']['rows_evaluated']}")
    print(f"nodes_evaluated: {report['counts']['nodes_evaluated']}")
    print(f"nodes_skipped:   {report['counts']['nodes_skipped']}")
    print(f"MAE(total_power):  {report['metrics']['mae_total_power']:.6f}")
    print(f"RMSE(total_power): {report['metrics']['rmse_total_power']:.6f}")
    print(f"detail_csv: {args.output_csv}")
    print(f"node_csv:   {args.output_node_csv}")
    print(f"report_json:{args.output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
