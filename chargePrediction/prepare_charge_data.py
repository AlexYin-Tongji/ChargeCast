"""Prepare charge data by aligning charge node_id with flow canonical nodes.

Canonical nodes follow the same policy used in flow training:
canonical_nodes = adj ∩ train ∩ test.

Default behavior:
- Filter in-place: overwrite data/charge/charge.csv
- Keep optional backup before overwrite
- Emit alignment report JSON
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowConfig.config as config
from src.data.alignment import build_alignment_report, build_canonical_nodes
from src.data.flow_data import FlowDataLoader


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def prepare_charge_data(
    charge_path: str,
    report_path: str,
    keep_backup: bool,
    backup_suffix: str,
    output_encoding: str,
) -> Dict[str, object]:
    loader = FlowDataLoader(config.DATA_DIR)

    train_df = loader.load_flow_data("train.csv")
    test_df = loader.load_flow_data("test.csv")
    adj_df = loader.load_adjacency("adj.csv")

    canonical_nodes, stats = build_canonical_nodes(
        adj_df,
        train_df,
        test_df,
        enabled=config.NODE_ALIGNMENT.get("enabled", True),
    )
    canonical_set = set(canonical_nodes)

    charge_df = pd.read_csv(charge_path)
    if "node_id" not in charge_df.columns:
        raise ValueError(f"`node_id` column not found in {charge_path}")

    charge_df["node_id"] = charge_df["node_id"].astype(str)
    charge_nodes: List[str] = sorted(charge_df["node_id"].dropna().unique().tolist())
    charge_set = set(charge_nodes)

    matched_nodes = sorted(list(charge_set & canonical_set))
    dropped_nodes = sorted(list(charge_set - canonical_set))

    filtered_df = charge_df[charge_df["node_id"].isin(canonical_set)].copy()

    backup_path = ""
    if keep_backup:
        backup_path = f"{charge_path}{backup_suffix}"
        shutil.copy2(charge_path, backup_path)

    # Overwrite original charge csv directly.
    filtered_df.to_csv(charge_path, index=False, encoding=output_encoding)

    flow_alignment_report = build_alignment_report(
        stats,
        preview_count=config.NODE_ALIGNMENT.get("dropped_preview_count", 20),
    )

    report = {
        "charge_path": charge_path,
        "backup_path": backup_path,
        "canonical_nodes_count": len(canonical_nodes),
        "charge_unique_nodes_count": len(charge_nodes),
        "matched_nodes_count": len(matched_nodes),
        "dropped_nodes_count": len(dropped_nodes),
        "charge_rows_before": int(len(charge_df)),
        "charge_rows_after": int(len(filtered_df)),
        "matched_nodes_preview": matched_nodes[:20],
        "dropped_nodes_preview": dropped_nodes[:20],
        "flow_alignment": flow_alignment_report,
        "output_encoding": output_encoding,
    }

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter charge.csv by flow canonical nodes and overwrite source.")
    parser.add_argument("--charge-path", default="data/charge/charge.csv", help="Input charge csv path.")
    parser.add_argument(
        "--report-path",
        default="chargePrediction/alignment_report.json",
        help="Path to save alignment report json.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV output encoding for overwritten file. Default utf-8-sig for Excel compatibility.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable creating backup file before overwrite.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak_before_filter",
        help="Backup suffix appended to charge csv path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = prepare_charge_data(
        charge_path=args.charge_path,
        report_path=args.report_path,
        keep_backup=not args.no_backup,
        backup_suffix=args.backup_suffix,
        output_encoding=args.encoding,
    )

    print("=" * 70)
    print("Charge-Flow Alignment Completed (In-place overwrite)")
    print("=" * 70)
    print(f"charge_path: {report['charge_path']}")
    print(f"backup_path: {report['backup_path']}")
    print(f"canonical_nodes_count: {report['canonical_nodes_count']}")
    print(f"charge_unique_nodes_count: {report['charge_unique_nodes_count']}")
    print(f"matched_nodes_count: {report['matched_nodes_count']}")
    print(f"dropped_nodes_count: {report['dropped_nodes_count']}")
    print(f"charge_rows_before: {report['charge_rows_before']}")
    print(f"charge_rows_after: {report['charge_rows_after']}")
    print(f"output_encoding: {report['output_encoding']}")
    print(f"dropped_nodes_preview: {report['dropped_nodes_preview']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
