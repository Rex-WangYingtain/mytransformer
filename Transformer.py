import torch
from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 device,
                 drop_prob=0.1):
        """
        :param src_pad_idx: 源向量pad索引
        :param trg_pad_idx: 目标向量pad索引
        :param enc_voc_size: 编码器词汇表大小
        :param dec_voc_size: 解码器词汇表大小
        :param d_model: embeding维度
        :param max_len: 序列最大长度
        :param n_heads: 头数
        :param ffn_hidden: 前馈神经网络隐藏层大小
        :param n_layers: 编解码器的层数
        :param device: 设备
        :param drop_prob: dropout概率
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            device=device,
            dropout=drop_prob
        )
        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            device=device,
            drop_prob=drop_prob
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        """
        pad掩码？
        """
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_q)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_k, 1)
        mask = q & k
        return mask
    
    def make_casual_mask(self, q, k):
        """
        因果掩码
        """
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out