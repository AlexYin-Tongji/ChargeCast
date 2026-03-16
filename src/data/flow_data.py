"""Flow data loading and sequence utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class FlowDataLoader:
    """Load long-format flow csv and adjacency csv."""

    def __init__(self, data_dir: str = "data/flow"):
        self.data_dir = data_dir

    def load_flow_data(self, filename: str = "train.csv") -> pd.DataFrame:
        """
        Load long-format flow data and pivot to wide format.

        Input columns:
            date, slice_start, hour_code, week_code, station_id, nev_flow
        Output columns:
            timestamp, time_key, <station_id_1>, <station_id_2>, ...
        """
        path = f"{self.data_dir}/{filename}"
        df = pd.read_csv(path)

        df["time_key"] = df["date"].astype(str) + " " + df["slice_start"].astype(str)

        pivot_df = (
            df.pivot_table(
                index="time_key",
                columns="station_id",
                values="nev_flow",
                aggfunc="sum",
            )
            .fillna(0)
            .reset_index()
        )

        unique_times = sorted(pivot_df["time_key"].unique())
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        pivot_df["timestamp"] = pivot_df["time_key"].map(time_to_idx).astype(int)

        cols = ["timestamp", "time_key"] + [
            c for c in pivot_df.columns if c not in ["timestamp", "time_key"]
        ]
        pivot_df = pivot_df[cols]

        # Keep node ids as strings for consistent alignment and topology handling.
        rename_map = {}
        for col in pivot_df.columns:
            if col not in {"timestamp", "time_key"}:
                rename_map[col] = str(col)
        pivot_df = pivot_df.rename(columns=rename_map)

        return pivot_df

    def load_adjacency(self, filename: str = "adj.csv") -> pd.DataFrame:
        """Load adjacency matrix with first column as source node id."""
        path = f"{self.data_dir}/{filename}"
        df = pd.read_csv(path, index_col=0)
        df.columns = [str(c) for c in df.columns]
        df.index = [str(i) for i in df.index]
        return df

    def get_nodes_from_flow(self, flow_df: pd.DataFrame) -> list:
        """Get station columns from wide-format flow dataframe."""
        numeric_cols = flow_df.select_dtypes(include=[np.number]).columns
        return [str(col) for col in numeric_cols if str(col) != "timestamp"]

    def get_nodes_from_adj(self, adj_df: pd.DataFrame) -> list:
        """Get all node ids from adjacency matrix in deterministic order."""
        source_nodes = [str(x) for x in adj_df.index.tolist()]
        target_nodes = [str(x) for x in adj_df.columns.tolist()]
        return list(dict.fromkeys(source_nodes + target_nodes))

    def get_exogenous_nodes(self, adj_df: pd.DataFrame) -> list:
        """Get exogenous nodes (in-degree == 0)."""
        nodes = self.get_nodes_from_adj(adj_df)
        in_degree = {}
        for node in nodes:
            if node in adj_df.columns:
                in_degree[node] = int((adj_df[node] > 0).sum())
            else:
                in_degree[node] = 0
        return [node for node in nodes if in_degree.get(node, 0) == 0]

    def get_endogenous_nodes(self, adj_df: pd.DataFrame) -> list:
        """Get endogenous nodes."""
        all_nodes = self.get_nodes_from_adj(adj_df)
        exogenous_nodes = set(self.get_exogenous_nodes(adj_df))
        return [n for n in all_nodes if n not in exogenous_nodes]


class FlowDataset(torch.utils.data.Dataset):
    """Generic dataset wrapper for flow sequences."""

    def __init__(self, flow_data: pd.DataFrame, node_indices: list, seq_len: int = 24, pred_len: int = 1):
        self.flow_data = flow_data
        self.node_indices = node_indices
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = self.flow_data.iloc[:, 1:].values
        if node_indices and max(node_indices) < self.data.shape[1]:
            self.data = self.data[:, node_indices]

        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0) + 1e-8
        self.normalized_data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.normalized_data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.normalized_data[idx : idx + self.seq_len]
        y = self.normalized_data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_sequences(data: np.ndarray, seq_len: int, pred_len: int):
    """Create rolling training samples from 2D [time, node] array."""
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(Y)
