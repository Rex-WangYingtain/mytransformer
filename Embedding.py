from torch import nn
import torch

class TokenEmbedding(nn.Embedding):
    """
    将输入的词汇表索引转换为知道维度的Embedding
    """
    def __init__(self, vocab_size, d_model):
        """
        :param vocab_size: 词汇表大小
        :param d_model: embeding维度
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        :param d_model: embeding维度
        :param max_len: 序列最大长度
        :param device: 设备
        """
        super(PositionEmbedding, self).__init__()

        # 初始化？
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        # 定义序列？
        pos = torch.arange(0, max_len, device=device)
        # 转化为浮点型张量
        pos = pos.float().unsqueeze(dim=1)

        # 生成序列并转为浮点型
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 根据奇数位置和偶数位置进行位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** ( _2i /d_model)))
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** ( _2i /d_model)))

    def forward(self, x):
        """
        前向传播
        """
        # 获取批量大小和序列长度
        batch_size, seq_len = x.size()
        # 返回编码矩阵中前seq_len的行数，也就是seq_learn和d_model大小的子矩阵
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb, pos_emb)
    