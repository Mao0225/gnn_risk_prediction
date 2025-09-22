import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class GNN(nn.Module):
    """
    图神经网络模型（带正则化），用于节点回归风险预测
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        """
        初始化 GNN 模型

        Args:
            in_channels: 输入特征维度（每个节点的特征数量）
            hidden_channels: 隐藏层维度（中间层特征数量）
            dropout: Dropout 概率（防止过拟合）
        """
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # 批标准化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GraphConv(hidden_channels, 1)  # 输出单值风险

    def forward(self, data):
        """
        前向传播

        Args:
            data: PyG 数据对象，包含 x, edge_index

        Returns:
            torch.Tensor: 节点预测值 [num_nodes]
        """
        x, edge_index = data.x, data.edge_index

        # 第一层图卷积 + 批标准化 + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 第二层图卷积输出
        x = self.conv2(x, edge_index)

        return x.squeeze(-1)  # [num_nodes]
