"""Node alignment utilities shared by train/eval scripts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _flow_node_columns(flow_df: pd.DataFrame) -> List[str]:
    numeric_cols = flow_df.select_dtypes(include=[np.number]).columns
    return [str(c) for c in numeric_cols if str(c) != "timestamp"]


def _adj_node_order(adj_df: pd.DataFrame) -> List[str]:
    source = [str(x) for x in adj_df.index.tolist()]
    target = [str(x) for x in adj_df.columns.tolist()]
    return _ordered_unique(source + target)


def summarize_dropped(nodes: List[str], preview_count: int = 20) -> Dict[str, object]:
    nodes = list(nodes)
    return {
        "count": len(nodes),
        "preview": nodes[:preview_count],
    }


def build_canonical_nodes(
    adj_df: pd.DataFrame,
    train_flow_df: pd.DataFrame,
    test_flow_df: pd.DataFrame,
    enabled: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Build canonical node list based on adj/train/test alignment policy."""
    adj_order = _adj_node_order(adj_df)
    adj_set = set(adj_order)

    train_nodes = _flow_node_columns(train_flow_df)
    test_nodes = _flow_node_columns(test_flow_df)
    train_set = set(train_nodes)
    test_set = set(test_nodes)

    if enabled:
        canonical_nodes = [n for n in adj_order if n in train_set and n in test_set]
    else:
        canonical_nodes = [n for n in adj_order if n in train_set]

    stats = {
        "adj_nodes": len(adj_set),
        "train_nodes": len(train_set),
        "test_nodes": len(test_set),
        "canonical_nodes": len(canonical_nodes),
        "dropped_from_adj": sorted(list(adj_set - set(canonical_nodes))),
        "dropped_from_train": sorted(list(train_set - set(canonical_nodes))),
        "dropped_from_test": sorted(list(test_set - set(canonical_nodes))),
    }
    return canonical_nodes, stats


def crop_adjacency(adj_df: pd.DataFrame, canonical_nodes: List[str]) -> pd.DataFrame:
    """Crop adjacency matrix to canonical nodes and keep fixed order."""
    canonical_nodes = [str(x) for x in canonical_nodes]
    return adj_df.loc[canonical_nodes, canonical_nodes].copy()


def align_flow_to_nodes(
    flow_df: pd.DataFrame,
    node_order: List[str],
    strict: bool = True,
) -> np.ndarray:
    """Convert flow dataframe into dense array ordered by node_order."""
    node_order = [str(x) for x in node_order]
    flow_cols = _flow_node_columns(flow_df)
    flow_set = set(flow_cols)

    if strict:
        missing = [n for n in node_order if n not in flow_set]
        if missing:
            raise ValueError(
                f"Flow data is missing {len(missing)} canonical nodes. Example: {missing[:5]}"
            )

    aligned = np.zeros((len(flow_df), len(node_order)), dtype=np.float32)
    for i, node in enumerate(node_order):
        if node in flow_df.columns:
            aligned[:, i] = flow_df[node].to_numpy(dtype=np.float32)
    return aligned


def build_alignment_report(stats: Dict[str, object], preview_count: int = 20) -> Dict[str, object]:
    """Compact report saved in checkpoints/logs."""
    return {
        "adj_nodes": stats["adj_nodes"],
        "train_nodes": stats["train_nodes"],
        "test_nodes": stats["test_nodes"],
        "canonical_nodes": stats["canonical_nodes"],
        "dropped_from_adj": summarize_dropped(stats["dropped_from_adj"], preview_count),
        "dropped_from_train": summarize_dropped(stats["dropped_from_train"], preview_count),
        "dropped_from_test": summarize_dropped(stats["dropped_from_test"], preview_count),
    }
