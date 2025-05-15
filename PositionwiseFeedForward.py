import torch.nn.functional as F
from torch import nn

class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model, hidden, dropout=0.1):
        """
        :param d_model: embeding维度
        :param hidden: 隐藏层大小
        :param dropout: dropout概率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc1 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x