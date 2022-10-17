import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class AddAndNorm(nn.Module):

    def __init__(self, size, dropout):
        super(AddAndNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddAndNorm(size, dropout), 2)
        self.size = size

    def forward(self, q, k, v, mask):
        x = self.sublayer[0](v, self.self_attn(q, k, v, mask))
        x = self.sublayer[1](x, self.feed_forward(x))
        return x


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer1 = clones(layer, N)
        self.layer2 = clones(layer, N)

    def forward(self, x_im, x_sk, mask):
        for layer1, layer2 in zip(self.layer1, self.layer2):
            # 在此交换Q
            # layer1 处理 sk
            x_sk1 = layer1(x_sk, x_im, x_sk, mask)
            # layer2 处理im
            x_im = layer2(x_im, x_sk, x_im, mask)
            x_sk = x_sk1
        return x_im, x_sk

    # def forward(self, x_sk, x_im, mask):
    #     for layer1, layer2 in zip(self.layer1, self.layer2):
    #         # 在此交换Q
    #         # layer1 处理 sk
    #         x_sk_output = layer1(x_im, x_sk, x_sk, mask)
    #         # layer2 处理im
    #         x_im_output = layer2(x_sk, x_im, x_im, mask)
    #     return x_sk_output, x_im_output


def attention(query, key, value, dropout=None, mask=None, pos=None):
    """
    dk = dv = dmodel/h = 64,h=8
    """
    d_k = query.size(-1)
    # query(batch,h,seq,dk)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 余弦相似度
    # scores = cos_similar(query, key)
    # relative_position_bias = pos.view(-1, 49, 49)  # nH, Wh*Ww, Wh*Ww
    # scores = scores + relative_position_bias.unsqueeze(0)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # 查看中间参数
    # print(p_attn.size(),torch.max(p_attn,dim=-1)[0])
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # self.pos = nn.Parameter(torch.zeros(h, 49 * 49))

    def forward(self, query, key, value, mask=None):
        """

        :param query: size(batch,seq,512)
        :param key:
        :param value:
        :param mask:
        :return:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        # size(batch,h,seq,dk)
        query, key, value = \
            [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for lin, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    d_model = 512
    d_ff = 2048 为论文中数值
    """

    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Cross_Attention(nn.Module):
    def __init__(self, args, h, n, d_model, d_ff, dropout):
        super(Cross_Attention, self).__init__()
        self.batch = args.batch
        self.args = args
        multi_head_attention = MultiHeadedAttention(h, d_model)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoderLayer = EncoderLayer(d_model, multi_head_attention, ffn, dropout)
        self.encoder = Encoder(encoderLayer, n)

    def forward(self, x, stage):
        length = x.size(0)
        x_sk = x[:length // 2]
        x_im = x[length // 2:]
        x_im, x_sk = self.encoder(x_im, x_sk, None)  # 不要mask
        return torch.cat((x_sk, x_im), dim=0)
        # x_sk_output, x_im_output = self.encoder(x_sk, x_im, None)  # 不要mask
        # return torch.cat((x_sk_output, x_im_output), dim=0)
