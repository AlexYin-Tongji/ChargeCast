import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM 流量预测模型"""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, pred_len: int = 1,
                 dropout: float = 0.1):
        """
        Args:
            input_size: 输入特征维度（节点数量）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            pred_len: 预测序列长度
            dropout: Dropout比例
        """
        super(LSTMPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出维度与输入特征数相同（每个节点一个预测值）
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch_size, seq_len, input_size)
        Returns:
            预测结果 (batch_size, input_size) - 每个输入特征对应一个预测
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时刻的输出
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层: 输出维度等于输入特征数量（每个节点一个预测）
        out = self.fc(out)  # (batch_size, input_size)

        return out


class MultiNodeLSTMPredictor(nn.Module):
    """多节点LSTM预测器，每个外生节点一个独立的LSTM"""

    def __init__(self, num_nodes: int, input_size: int = 1,
                 hidden_size: int = 32, num_layers: int = 2,
                 pred_len: int = 1, dropout: float = 0.1):
        """
        Args:
            num_nodes: 外生节点数量
            input_size: 每个节点的输入维度（通常为1）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            pred_len: 预测序列长度
            dropout: Dropout比例
        """
        super(MultiNodeLSTMPredictor, self).__init__()

        self.num_nodes = num_nodes

        # 为每个节点创建独立的LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            for _ in range(num_nodes)
        ])

        # 每个节点独立的全连接层
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, pred_len)
            for _ in range(num_nodes)
        ])

    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch_size, num_nodes, seq_len)
        Returns:
            预测结果 (batch_size, num_nodes, pred_len)
        """
        batch_size = x.size(0)
        outputs = []

        # 对每个节点分别进行预测
        for node_idx in range(self.num_nodes):
            # 获取该节点的输入 (batch_size, seq_len, 1)
            node_input = x[:, node_idx, :].unsqueeze(-1)

            # LSTM
            lstm_out, (h_n, c_n) = self.lstm_layers[node_idx](node_input)

            # 取最后一个时刻的输出
            out = lstm_out[:, -1, :]

            # 全连接层
            out = self.fc_layers[node_idx](out)
            outputs.append(out)

        # 合并所有节点的预测结果
        output = torch.stack(outputs, dim=1)  # (batch_size, num_nodes, pred_len)

        return output


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            output = model(X)
            loss = criterion(output, Y.squeeze(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)
