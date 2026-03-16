"""
传播概率模块
使用神经网络 + Beta 分布建模

输入: 前 L 个时间步的流量
输出: Beta 分布的 alpha, beta 参数
更新: 贝叶斯推断 (后验 = 先验 + 数据)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import flowConfig.config as config


class BayesianPropagation(nn.Module):
    """贝叶斯传播概率

    先验: Beta(alpha_0, beta_0)，其中 alpha_0, beta_0 来自 prior_prob 和 prior_precision
    后验: alpha = alpha_0 + delta_alpha, beta = beta_0 + delta_beta
    """

    def __init__(self, num_nodes: int, node_ids: list):
        super(BayesianPropagation, self).__init__()

        self.num_nodes = num_nodes
        self.node_ids = node_ids
        self.node_to_idx = {n: i for i, n in enumerate(node_ids)}

        input_window = config.PROPAGATION["input_window"]
        hidden_dim = config.PROPAGATION["hidden_dim"]
        num_layers = config.PROPAGATION["num_layers"]

        # 从先验概率转换到 alpha_0, beta_0
        prior_prob = config.PROPAGATION["prior_prob"]
        precision = config.PROPAGATION["prior_precision"]

        self.alpha_0 = prior_prob * precision
        self.beta_0 = (1 - prior_prob) * precision

        print(f"传播概率先验: alpha_0={self.alpha_0}, beta_0={self.beta_0}, E[p]={prior_prob}")

        # 每个节点独特的网络：预测增量 delta_alpha, delta_beta
        # 使用索引作为 key，避免节点ID中的特殊字符问题
        self.node_networks = nn.ModuleList()

        for i, node_id in enumerate(node_ids):
            layers = []
            in_dim = input_window

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim

            # 输出 delta_alpha 和 delta_beta（可正可负）
            layers.append(nn.Linear(in_dim, 2))

            self.node_networks.append(nn.Sequential(*layers))

    def forward(self, node_id, flow_history: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回传播概率 p

        Args:
            node_id: 节点ID（字符串或索引）
            flow_history: 前L个时间步的流量 [batch_size, L] 或 [L]

        Returns:
            传播概率 p [batch_size]
        """
        if flow_history.dim() == 1:
            flow_history = flow_history.unsqueeze(0)

        # 使用索引访问网络
        idx = self.node_to_idx[node_id] if node_id in self.node_to_idx else node_id
        net = self.node_networks[idx]
        delta = net(flow_history)  # [batch, 2]

        delta_alpha = delta[:, 0]
        delta_beta = delta[:, 1]

        # 后验 = 先验 + 数据
        # 使用 softplus 确保增量非负（单调增加）
        alpha = self.alpha_0 + F.softplus(delta_alpha)
        beta = self.beta_0 + F.softplus(delta_beta)

        # 传播概率 = Beta 分布均值
        p = alpha / (alpha + beta)

        return p

    def get_params(self, node_id, flow_history: torch.Tensor) -> tuple:
        """
        获取 Beta 分布参数

        Returns:
            (alpha, beta, p)
        """
        if flow_history.dim() == 1:
            flow_history = flow_history.unsqueeze(0)

        idx = self.node_to_idx[node_id] if node_id in self.node_to_idx else node_id
        net = self.node_networks[idx]
        delta = net(flow_history)

        alpha = self.alpha_0 + F.softplus(delta[:, 0])
        beta = self.beta_0 + F.softplus(delta[:, 1])
        p = alpha / (alpha + beta)

        return alpha, beta, p

    def forward_all(self, flow_history: torch.Tensor) -> torch.Tensor:
        """
        批量处理所有节点

        Args:
            flow_history: [batch_size, num_nodes, L]

        Returns:
            传播概率 [batch_size, num_nodes]
        """
        batch_size = flow_history.shape[0]
        device = flow_history.device

        outputs = []

        for i, node_id in enumerate(self.node_ids):
            flow = flow_history[:, i, :]  # [batch, L]
            p = self.forward(node_id, flow)
            outputs.append(p)

        return torch.stack(outputs, dim=1)


class SimplifiedPropagation(nn.Module):
    """简化版传播概率（直接预测概率值，加先验约束）"""

    def __init__(self, num_nodes: int, node_ids: list):
        super(SimplifiedPropagation, self).__init__()

        self.num_nodes = num_nodes
        self.node_ids = node_ids
        self.node_to_idx = {n: i for i, n in enumerate(node_ids)}

        input_window = config.PROPAGATION["input_window"]
        hidden_dim = config.PROPAGATION["hidden_dim"]

        # 先验
        self.prior_prob = config.PROPAGATION["prior_prob"]

        # 每个节点一个网络（使用 ModuleList）
        self.node_layers = nn.ModuleList()

        for node_id in node_ids:
            self.node_layers.append(nn.Sequential(
                nn.Linear(input_window, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ))

    def forward(self, node_id, flow_history: torch.Tensor) -> torch.Tensor:
        """
        输出概率 p in [0, 1]
        """
        if flow_history.dim() == 1:
            flow_history = flow_history.unsqueeze(0)

        idx = self.node_to_idx[node_id] if node_id in self.node_to_idx else node_id
        layer = self.node_layers[idx]
        logit = layer(flow_history).squeeze(-1)

        # sigmoid 限制到 [0,1]
        p = torch.sigmoid(logit)

        return p


def demo():
    """测试传播概率"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids = [1001, 1002, 1003, 1004, 1005]
    num_nodes = len(node_ids)

    L = config.PROPAGATION["input_window"]

    prop = BayesianPropagation(num_nodes, node_ids).to(device)

    # 模拟输入
    flow_history = torch.randn(4, L).abs().to(device)

    print("输入历史流量:", flow_history.shape)

    for node_id in node_ids[:2]:
        p = prop(node_id, flow_history)
        alpha, beta, _ = prop.get_params(node_id, flow_history)
        print(f"节点 {node_id}: p={p[0].item():.4f}, alpha={alpha[0].item():.4f}, beta={beta[0].item():.4f}")


if __name__ == "__main__":
    demo()
