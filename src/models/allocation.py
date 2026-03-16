"""
分配概率模块
使用傅里叶级数拟合 Dirichlet alpha 参数

alpha(t) = alpha0 + sum_k [ak * sin(k*w*t) + bk * cos(k*w*t)]
P_ij = alpha_ij / sum_j(alpha_ij)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import math

import flowConfig.config as config


class FourierAllocation(nn.Module):
    """傅里叶级数拟合分配概率 alpha

    每条边 (上游 -> 下游) 有独特的参数
    alpha(t) = a0 + sum_k [ak * sin(k*w*t) + bk * cos(k*w*t)]
    """

    def __init__(self, num_edges: int, num_frequencies: int = None):
        """
        Args:
            num_edges: 边的数量
            num_frequencies: 傅里叶频率数量 K
        """
        super(FourierAllocation, self).__init__()

        if num_frequencies is None:
            num_frequencies = config.ALLOCATION["num_frequencies"]

        self.num_edges = num_edges
        self.num_frequencies = num_frequencies

        # 傅里叶参数
        # alpha(t) = a0 + sum_k [ak * sin(k*w*t) + bk * cos(k*w*t)]
        # 每条边有: 1个a0, K个ak, K个bk

        # 直流分量 a0
        self.alpha0 = nn.Parameter(
            torch.ones(num_edges) * config.ALLOCATION["prior_alpha0"]
        )

        # 傅里叶系数 ak, bk
        self.ak = nn.Parameter(
            torch.randn(num_edges, num_frequencies) * config.ALLOCATION["prior_ak_bk"]
        )
        self.bk = nn.Parameter(
            torch.randn(num_edges, num_frequencies) * config.ALLOCATION["prior_ak_bk"]
        )

        # 角频率
        self.omega = config.ALLOCATION["omega"]

    def forward(self, hour: torch.Tensor) -> torch.Tensor:
        """
        计算每条边的 alpha 值

        Args:
            hour: 小时 [batch_size] 或标量

        Returns:
            alpha 值 [batch_size, num_edges]
        """
        # 确保 hour 是一维
        if hour.dim() == 0:
            hour = hour.unsqueeze(0)

        batch_size = hour.shape[0]
        device = self.alpha0.device

        # 归一化小时到 [0, 2*pi]
        hour_rad = hour * self.omega  # [batch_size]

        # 计算傅里叶项
        alpha = self.alpha0.unsqueeze(0).expand(batch_size, -1)  # [batch, num_edges]

        for k in range(self.num_frequencies):
            k_freq = k + 1
            # sin(k*w*t) 和 cos(k*w*t)
            sin_term = torch.sin(k_freq * hour_rad)  # [batch]
            cos_term = torch.cos(k_freq * hour_rad)  # [batch]

            # 获取第 k 个频率的系数: [num_edges]
            ak_k = self.ak[:, k]  # [num_edges]
            bk_k = self.bk[:, k]  # [num_edges]

            # 扩展到 [batch, num_edges]
            sin_exp = sin_term.unsqueeze(-1) * ak_k.unsqueeze(0)  # [batch, num_edges]
            cos_exp = cos_term.unsqueeze(-1) * bk_k.unsqueeze(0)  # [batch, num_edges]

            # 累加
            alpha = alpha + sin_exp + cos_exp

        # 确保 alpha > 0
        alpha = torch.clamp(alpha, min=1e-6)

        return alpha

    def get_probability(self, hour: torch.Tensor) -> torch.Tensor:
        """
        获取分配概率（归一化后的 P_ij）

        Args:
            hour: 小时 [batch_size]

        Returns:
            分配概率 P_ij [batch_size, num_edges]
        """
        alpha = self.forward(hour)  # [batch, num_edges]

        # 归一化：每条边对应的目标节点
        # 这里需要知道每条边的目标节点
        # 暂时返回 alpha，后续在 EdgeAllocation 中处理

        return alpha


class EdgeAllocation(nn.Module):
    """边级别的分配概率

    对于每个上游节点 i，将其流量分配到所有下游节点
    每条边 (i -> j) 有独特的 alpha 参数
    """

    def __init__(self, upstream_nodes: list, downstream_nodes_dict: dict):
        """
        Args:
            upstream_nodes: 上游节点列表
            downstream_nodes_dict: {上游节点: [下游节点列表]}
        """
        super(EdgeAllocation, self).__init__()

        self.upstream_nodes = upstream_nodes
        self.downstream_nodes_dict = downstream_nodes_dict

        # 构建边列表和映射
        self.edges = []  # [(upstream, downstream), ...]
        self.upstream_to_edges = {}  # upstream -> 边索引列表

        for up_node in upstream_nodes:
            down_nodes = downstream_nodes_dict.get(up_node, [])
            edge_indices = []
            for down_node in down_nodes:
                edge_idx = len(self.edges)
                self.edges.append((up_node, down_node))
                edge_indices.append(edge_idx)
            self.upstream_to_edges[up_node] = edge_indices

        self.num_edges = len(self.edges)

        # 傅里叶分配网络
        self.fourier_alloc = FourierAllocation(self.num_edges)

    def forward(self, hour: torch.Tensor) -> dict:
        """
        计算所有边的分配概率

        Args:
            hour: 小时 [batch_size]

        Returns:
            {上游节点: [分配概率列表]}
        """
        alpha = self.fourier_alloc(hour)  # [batch, num_edges]

        results = {}

        for up_node in self.upstream_nodes:
            edge_indices = self.upstream_to_edges.get(up_node, [])

            if not edge_indices:
                results[up_node] = None
                continue

            # 获取该上游节点对应的 alpha
            alpha_up = alpha[:, edge_indices]  # [batch, num_downstream]

            # 归一化得到概率
            p_alloc = alpha_up / (alpha_up.sum(dim=1, keepdim=True) + 1e-8)

            results[up_node] = p_alloc

        return results


class SimplifiedAllocation(nn.Module):
    """简化版分配概率（每个上游节点统一分配）

    对于每个上游节点 i，其所有下游使用相同的分配参数
    """

    def __init__(self, num_nodes: int, node_ids: list):
        super(SimplifiedAllocation, self).__init__()

        self.num_nodes = num_nodes
        self.node_ids = node_ids
        self.node_to_idx = {n: i for i, n in enumerate(node_ids)}

        num_frequencies = config.ALLOCATION["num_frequencies"]

        # 每个节点一套傅里叶参数
        self.alpha0 = nn.Parameter(
            torch.ones(num_nodes) * config.ALLOCATION["prior_alpha0"]
        )
        self.ak = nn.Parameter(
            torch.randn(num_nodes, num_frequencies) * config.ALLOCATION["prior_ak_bk"]
        )
        self.bk = nn.Parameter(
            torch.randn(num_nodes, num_frequencies) * config.ALLOCATION["prior_ak_bk"]
        )

        self.omega = config.ALLOCATION["omega"]
        self.num_frequencies = num_frequencies

    def forward(self, node_ids: list, hour: torch.Tensor) -> torch.Tensor:
        """
        获取分配概率

        Args:
            node_ids: 节点ID列表 [num_nodes]
            hour: 小时 [batch_size]

        Returns:
            分配概率 [batch_size, num_nodes, max_downstream]
            需要外部根据下游节点数量进行处理
        """
        batch_size = hour.shape[0]
        device = hour.device

        hour_rad = hour * self.omega

        # 计算每个节点的 alpha
        alpha = self.alpha0.unsqueeze(0).expand(batch_size, -1).clone()

        for k in range(self.num_frequencies):
            k_freq = k + 1
            sin_term = torch.sin(k_freq * hour_rad)
            cos_term = torch.cos(k_freq * hour_rad)

            alpha = alpha + self.ak.unsqueeze(0) * sin_term.unsqueeze(-1) \
                          + self.bk.unsqueeze(0) * cos_term.unsqueeze(-1)

        alpha = torch.clamp(alpha, min=1e-6)

        return alpha


def demo():
    """测试傅里叶分配"""
    num_edges = 5
    alloc = FourierAllocation(num_edges)

    # 测试不同时间点
    hours = torch.tensor([0.0, 6.0, 12.0, 18.0])
    alpha = alloc(hours)

    print("小时:", hours.numpy())
    print("Alpha shape:", alpha.shape)
    print("Alpha values:\n", alpha.detach().numpy())


if __name__ == "__main__":
    demo()
