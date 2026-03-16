"""Main flow predictor with exogenous LSTM + allocation + propagation."""

import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn

import flowConfig.config as config
from .allocation import FourierAllocation
from .lstm import MultiNodeLSTMPredictor
from .propagation import BayesianPropagation
from .topology import TopologyProcessor


class FlowPredictor(nn.Module):
    """End-to-end flow predictor."""

    def __init__(
        self,
        num_nodes: int,
        node_ids: List[str],
        exogenous_nodes: List[str],
        topo_order: List[str],
        upstream_dict: Dict[str, List[str]],
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_ids = node_ids
        self.exogenous_nodes = exogenous_nodes
        self.topo_order = topo_order
        self.upstream_dict = upstream_dict

        self.node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
        self.idx_to_node = {idx: node for idx, node in enumerate(node_ids)}
        self.exo_set = set(exogenous_nodes)
        self.exo_indices = [self.node_to_idx[n] for n in exogenous_nodes]
        self.endo_indices = [self.node_to_idx[n] for n in node_ids if n not in self.exo_set]

        self.exogenous_lstm = MultiNodeLSTMPredictor(
            num_nodes=len(exogenous_nodes),
            input_size=1,
            hidden_size=config.LSTM["hidden_size"],
            num_layers=config.LSTM["num_layers"],
            pred_len=1,
            dropout=config.LSTM.get("dropout", 0.0),
        )

        self.edge_list = []
        self.downstream_edges = {}
        self.edge_lookup = {}

        for down_node in node_ids:
            up_nodes = upstream_dict.get(down_node, [])
            for up_node in up_nodes:
                up_idx = self.node_to_idx[up_node]
                down_idx = self.node_to_idx[down_node]
                edge_idx = len(self.edge_list)
                self.edge_list.append((up_idx, down_idx))
                self.edge_lookup[(up_idx, down_idx)] = edge_idx
                self.downstream_edges.setdefault(up_idx, []).append((down_idx, edge_idx))

        self.num_edges = len(self.edge_list)
        self.allocation = FourierAllocation(self.num_edges)
        self.propagation = BayesianPropagation(num_nodes, node_ids)
        gaussian_cfg = getattr(config, "ENDOGENOUS_GAUSSIAN", {})
        self.endogenous_gaussian_enabled = bool(gaussian_cfg.get("enabled", False))
        self.endogenous_gaussian_per_node = bool(gaussian_cfg.get("per_node", True))
        init_mu = float(gaussian_cfg.get("init_mu", 0.0))
        init_log_var = float(gaussian_cfg.get("init_log_var", -4.0))

        if self.endogenous_gaussian_per_node:
            self.endo_noise_mu = nn.Parameter(torch.full((self.num_nodes,), init_mu))
            self.endo_noise_log_var = nn.Parameter(torch.full((self.num_nodes,), init_log_var))
        else:
            self.endo_noise_mu = nn.Parameter(torch.tensor(init_mu))
            self.endo_noise_log_var = nn.Parameter(torch.tensor(init_log_var))

        self.register_buffer("norm_mean", torch.zeros(self.num_nodes))
        self.register_buffer("norm_std", torch.ones(self.num_nodes))
        self._has_norm_stats = False

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Attach train normalization stats for physical-space constraints."""
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.norm_mean.device).view(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=self.norm_std.device).view(-1)
        if mean_t.numel() != self.num_nodes or std_t.numel() != self.num_nodes:
            raise ValueError(
                f"Normalization size mismatch: expected {self.num_nodes}, got "
                f"mean={mean_t.numel()}, std={std_t.numel()}."
            )
        self.norm_mean.copy_(mean_t)
        self.norm_std.copy_(std_t)
        self._has_norm_stats = True

    def _to_physical(self, x_norm: torch.Tensor, node_idx: int) -> torch.Tensor:
        return x_norm * self.norm_std[node_idx] + self.norm_mean[node_idx]

    def _to_normalized(self, x_phys: torch.Tensor, node_idx: int) -> torch.Tensor:
        return (x_phys - self.norm_mean[node_idx]) / (self.norm_std[node_idx] + 1e-8)

    def _apply_endogenous_gaussian(self, x_next: torch.Tensor) -> torch.Tensor:
        """
        Apply learned Gaussian compensation on endogenous node outputs.
        Train mode: sample using reparameterization.
        Eval mode: optionally add deterministic mean compensation.
        """
        if not self.endogenous_gaussian_enabled or not self.endo_indices:
            return x_next

        cfg = config.ENDOGENOUS_GAUSSIAN
        use_sampling = self.training and bool(cfg.get("train_use_sampling", True))
        eval_use_mean = bool(cfg.get("eval_use_mean", True))

        if self.endogenous_gaussian_per_node:
            mu = self.endo_noise_mu[self.endo_indices].unsqueeze(0)  # [1, E]
            log_var = self.endo_noise_log_var[self.endo_indices].unsqueeze(0)  # [1, E]
        else:
            mu = self.endo_noise_mu.view(1, 1).expand(1, len(self.endo_indices))
            log_var = self.endo_noise_log_var.view(1, 1).expand(1, len(self.endo_indices))

        if use_sampling:
            eps = torch.randn(x_next.shape[0], len(self.endo_indices), device=x_next.device, dtype=x_next.dtype)
            std = torch.exp(0.5 * log_var).expand_as(eps)
            delta = mu.expand_as(eps) + std * eps
        elif eval_use_mean:
            delta = mu.expand(x_next.shape[0], -1)
        else:
            return x_next

        x_next = x_next.clone()
        x_next[:, self.endo_indices] = x_next[:, self.endo_indices] + delta
        return x_next

    def predict_exogenous(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Predict next-step values for all exogenous nodes from history."""
        if not self.exo_indices:
            return x_hist.new_zeros((x_hist.shape[0], 0))

        exo_data = x_hist[:, :, self.exo_indices]  # [B, T, E]
        exo_data = exo_data.transpose(1, 2)  # [B, E, T]
        return self.exogenous_lstm(exo_data).squeeze(-1)  # [B, E]

    def _allocation_edge_prob(self, alpha: torch.Tensor, up_idx: int, edge_idx: int) -> torch.Tensor:
        """Normalize alpha over all outgoing edges from one upstream node."""
        out_edges = self.downstream_edges.get(up_idx, [])
        if not out_edges:
            return alpha[:, edge_idx]

        denom = torch.zeros(alpha.shape[0], device=alpha.device)
        for _, idx in out_edges:
            denom = denom + alpha[:, idx]
        denom = denom + 1e-8
        return alpha[:, edge_idx] / denom

    def predict_step(
        self,
        x_current: torch.Tensor,
        x_prev: torch.Tensor,
        hour: torch.Tensor,
        flow_history: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one step for all nodes."""
        batch_size = x_current.shape[0]
        device = x_current.device

        output = x_current.clone()
        alpha = self.allocation(hour)  # [B, num_edges]
        use_physical_non_negative = (
            config.OUTPUT_CONSTRAINTS.get("physical_non_negative", False) and self._has_norm_stats
        )

        for node in self.topo_order:
            if node in self.exo_set:
                continue

            node_idx = self.node_to_idx[node]
            up_nodes = self.upstream_dict.get(node, [])

            if isinstance(up_nodes, (int, float, np.integer, np.floating)):
                up_nodes = [int(up_nodes)] if up_nodes != 0 else []
            elif not isinstance(up_nodes, list):
                try:
                    up_nodes = list(up_nodes)
                except Exception:
                    up_nodes = []

            if not up_nodes:
                continue

            total_contribution = torch.zeros(batch_size, device=device)

            for up_node in up_nodes:
                up_idx = self.node_to_idx[up_node]
                edge_idx = self.edge_lookup.get((up_idx, node_idx))
                if edge_idx is None:
                    continue

                p_alloc = self._allocation_edge_prob(alpha, up_idx, edge_idx)
                flow_hist_up = flow_history[:, up_idx, :]
                p_prop = self.propagation(up_node, flow_hist_up)

                x_up_current = x_current[:, up_idx]
                x_up_prev = x_prev[:, up_idx]

                if use_physical_non_negative:
                    x_up_current = torch.relu(self._to_physical(x_up_current, up_idx))
                    x_up_prev = torch.relu(self._to_physical(x_up_prev, up_idx))

                contribution = p_alloc * (p_prop * x_up_current + (1.0 - p_prop) * x_up_prev)
                total_contribution = total_contribution + contribution

            if use_physical_non_negative:
                total_contribution = torch.relu(total_contribution)
                output[:, node_idx] = self._to_normalized(total_contribution, node_idx)
            else:
                output[:, node_idx] = total_contribution

        return output

    def forward(
        self,
        x_hist: torch.Tensor,
        hours: torch.Tensor,
        exogenous_truth: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Multi-step prediction.

        Args:
            x_hist: [B, seq_len, num_nodes]
            hours: [B, pred_len]
            exogenous_truth: optional [B, pred_len, num_nodes]
            teacher_forcing_ratio: probability of using exogenous ground truth.
        """
        batch_size = x_hist.shape[0]
        pred_len = hours.shape[1]

        hist_len = config.PROPAGATION["input_window"]

        if x_hist.shape[1] >= hist_len:
            flow_history = x_hist[:, -hist_len:, :].transpose(1, 2)  # [B, N, L]
        else:
            pad = torch.zeros(
                batch_size,
                hist_len - x_hist.shape[1],
                self.num_nodes,
                dtype=x_hist.dtype,
                device=x_hist.device,
            )
            padded = torch.cat([pad, x_hist], dim=1)
            flow_history = padded.transpose(1, 2)

        history_window = x_hist.clone()
        x_current = x_hist[:, -1, :].clone()
        x_prev = x_hist[:, -2, :].clone() if x_hist.shape[1] > 1 else x_hist[:, -1, :].clone()

        outputs = []

        for t in range(pred_len):
            # Re-compute exogenous prediction every step from rolling history.
            if self.exo_indices:
                exo_pred = self.predict_exogenous(history_window)

                if exogenous_truth is not None and teacher_forcing_ratio > 0.0:
                    exo_truth_t = exogenous_truth[:, t, self.exo_indices]
                    mask = torch.rand_like(exo_pred) < teacher_forcing_ratio
                    exo_used = torch.where(mask, exo_truth_t, exo_pred)
                else:
                    exo_used = exo_pred

                if config.OUTPUT_CONSTRAINTS.get("physical_non_negative", False) and self._has_norm_stats:
                    for i, exo_idx in enumerate(self.exo_indices):
                        exo_phys = torch.relu(self._to_physical(exo_used[:, i], exo_idx))
                        exo_used[:, i] = self._to_normalized(exo_phys, exo_idx)

                x_current[:, self.exo_indices] = exo_used

            # Update propagation history by the current state.
            flow_history = torch.cat([flow_history[:, :, 1:], x_current.unsqueeze(-1)], dim=2)

            hour_t = hours[:, t]
            x_next = self.predict_step(x_current, x_prev, hour_t, flow_history)
            x_next = self._apply_endogenous_gaussian(x_next)
            outputs.append(x_next)

            # Roll temporal states for next step.
            x_prev = x_current
            x_current = x_next
            history_window = torch.cat([history_window[:, 1:, :], x_next.unsqueeze(1)], dim=1)

        return torch.stack(outputs, dim=1)


def build_predictor(data_dir: str = None) -> tuple:
    """Build predictor from adjacency data."""
    from src.data.flow_data import FlowDataLoader

    if data_dir is None:
        data_dir = config.DATA_DIR

    loader = FlowDataLoader(data_dir)
    adj_df = loader.load_adjacency("adj.csv")

    topo = TopologyProcessor(adj_df)

    exogenous_nodes = list(topo.exogenous_nodes)
    topo_order = topo.get_topological_order()
    all_nodes = list(topo_order)

    upstream_dict = {}
    for node in all_nodes:
        upstream_dict[node] = topo.get_upstream_nodes(node)

    model = FlowPredictor(
        num_nodes=len(all_nodes),
        node_ids=all_nodes,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=upstream_dict,
    )

    return model, topo


def demo():
    """Lightweight demo."""
    node_ids = ["1001", "1002", "1003", "1004", "1005"]
    exogenous_nodes = ["1001", "1002"]
    topo_order = ["1001", "1002", "1003", "1004", "1005"]
    upstream_dict = {
        "1001": [],
        "1002": [],
        "1003": ["1001"],
        "1004": ["1001", "1002"],
        "1005": ["1003", "1004"],
    }

    model = FlowPredictor(
        num_nodes=len(node_ids),
        node_ids=node_ids,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=upstream_dict,
    )

    batch_size = 4
    seq_len = 24
    x_hist = torch.randn(batch_size, seq_len, len(node_ids))
    hours = torch.randint(0, 24, (batch_size, 1)).float()

    output = model(x_hist, hours)

    print("x_hist:", x_hist.shape)
    print("hours:", hours.shape)
    print("output:", output.shape)
    print("params:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    demo()
