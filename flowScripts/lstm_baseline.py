"""
LSTM Baseline
为每个站点训练一个独立的LSTM模型，计算所有站点上的平均loss
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import flowConfig.config as config
from src.data.flow_data import FlowDataLoader, create_sequences


class SingleNodeLSTM(nn.Module):
    """单节点LSTM预测器"""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        super(SingleNodeLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]  # [batch, hidden_size]
        out = self.fc(out)  # [batch, 1]
        return out


def train_node_lstm(X, Y, node_idx, device, epochs=50, batch_size=32, lr=0.001):
    """为单个节点训练LSTM"""
    # 提取该节点的数据
    X_node = X[:, :, node_idx]  # [batch, seq_len]
    Y_node = Y[:, 0, node_idx]  # [batch]

    # 转换为 tensor
    X_tensor = torch.FloatTensor(X_node).unsqueeze(-1).to(device)  # [batch, seq_len, 1]
    Y_tensor = torch.FloatTensor(Y_node).unsqueeze(-1).to(device)  # [batch, 1]

    # 创建模型
    model = SingleNodeLSTM(
        input_size=1,
        hidden_size=config.LSTM["hidden_size"],
        num_layers=config.LSTM["num_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor)
        loss = criterion(pred, Y_tensor).item()

    return model, loss


def evaluate_node_lstm(model, X, Y, node_idx, device):
    """评估单节点LSTM"""
    X_node = X[:, :, node_idx]
    Y_node = Y[:, 0, node_idx]

    X_tensor = torch.FloatTensor(X_node).unsqueeze(-1).to(device)
    Y_tensor = torch.FloatTensor(Y_node).unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_tensor)
        criterion = nn.MSELoss()
        mse = criterion(pred, Y_tensor).item()

        # 也计算 MAE
        mae_criterion = nn.L1Loss()
        mae = mae_criterion(pred, Y_tensor).item()

    return mse, mae


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    # 加载数据
    loader = FlowDataLoader(config.DATA_DIR)

    # 加载训练数据（用于训练）
    train_data = loader.load_flow_data("train.csv")
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    data_cols = [col for col in numeric_cols if col != 'timestamp']
    train_raw = train_data[data_cols].values

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-8
    train_norm = (train_raw - mean) / std

    SEQ_LEN = config.SEQ_LEN
    PRED_LEN = config.PRED_LEN

    X_train, Y_train = create_sequences(train_norm, SEQ_LEN, PRED_LEN)

    # 加载测试数据
    test_data = loader.load_flow_data("test.csv")
    test_raw = test_data[data_cols].values
    test_norm = (test_raw - mean) / std
    X_test, Y_test = create_sequences(test_norm, SEQ_LEN, PRED_LEN)

    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    num_nodes = X_train.shape[2]
    print(f"\nTraining LSTM for {num_nodes} nodes...")

    # 为每个节点训练LSTM
    total_mse = 0
    total_mae = 0

    for node_idx in range(num_nodes):
        if node_idx % 100 == 0:
            print(f"Processing node {node_idx}/{num_nodes}...")

        # 训练
        model, train_loss = train_node_lstm(
            X_train, Y_train, node_idx, device,
            epochs=20, batch_size=64, lr=0.01
        )

        # 测试
        mse, mae = evaluate_node_lstm(model, X_test, Y_test, node_idx, device)

        total_mse += mse
        total_mae += mae

    # 平均
    avg_mse = total_mse / num_nodes
    avg_mae = total_mae / num_nodes

    print(f"\n{'='*50}")
    print(f"LSTM Baseline Results (per node):")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
