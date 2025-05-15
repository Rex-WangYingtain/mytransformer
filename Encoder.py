from torch import nn
import torch.nn.functional as F
import MutiHeadAttention
import LayerNorm
import Embedding
from PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MutiHeadAttention.MutiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)  # 残差链接
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = Embedding.TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
    