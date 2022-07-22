import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Graph WaveNet ST-block
######################################################################
class GW(nn.Module):
    def __init__(self, K, supports, nodevec1, nodevec2, c_in, c_out, kernel_size):
        super(GW, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv1= CausalConv2d(c_in, c_out, kernel_size, 1, dilation=1)
        self.gate_conv1 = CausalConv2d(c_in, c_out, kernel_size, 1, dilation=1)
        self.filter_conv2 = CausalConv2d(c_in, c_out, kernel_size, 1, dilation=2)
        self.gate_conv2 = CausalConv2d(c_in, c_out, kernel_size, 1, dilation=2)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

        c_in = (K * (len(supports) + 1) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp1 = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32
        self.c_out = c_out
        self.K = K
        self.supports = supports
        self.nconv = nconv()

        self.mlp2 = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32

    def forward(self, x):
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))  # 差别不大
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        new_supports = self.supports + [adp]

        # x = self.relu(x)

        filter1 = (self.filter_conv1(x))
        gate1 = torch.sigmoid(self.gate_conv1(x))
        x = filter1 * gate1

        out = [x]
        for a in new_supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        x = self.mlp1(h)

        filter2 = (self.filter_conv2(x))
        gate2 = torch.sigmoid(self.gate_conv2(x))
        x = filter2 * gate2

        out = [x]
        for a in new_supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        x = self.mlp2(h)

        # output = self.bn(x)

        return x


class CNN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]这些block的input必须具有相同的shape？
        :return:
        """
        x = self.relu(x)
        output = (self.filter_conv(x))
        output = self.bn(output)

        return output


class Cheb_gcn(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        """
        :param K: K-order
        :param cheb_polynomials: laplacian matrix？
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(Cheb_gcn, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k])
                T_k = alpha * self.cheb_polynomials[k] + beta * adp

                # T_k = self.cheb_polynomials[k]  # [N, N]
                self.theta_k = self.Theta[k]  # [c_in, c_out]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)
            outputs.append(output.unsqueeze(-1))
        outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        outputs = self.bn(outputs)

        return outputs


######################################################################
# DCRNN ST-block
######################################################################
class DCRNN(nn.Module):
    def __init__(self, K, supports, nodevec1, nodevec2, c_in, c_out):
        super(DCRNN, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.gru1 = GRU(c_in, c_out)
        self.gru2 = GRU(c_in, c_out)

        c_in = (K * (len(supports) + 1) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp1 = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32
        self.c_out = c_out
        self.K = K
        self.supports = supports
        self.nconv = nconv()
        self.mlp2 = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32

    def forward(self, x):
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))  # 差别不大
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        new_supports = self.supports + [adp]

        # x = self.relu(x)

        x = self.gru1(x)
        out = [x]
        for a in new_supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        x = self.mlp1(h)

        x = self.gru2(x)
        out = [x]
        for a in new_supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        x = self.mlp2(h)

        # output = self.bn(x)

        return x


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, hiddens, input_size, activation_function, out_act, dropout_ratio=0.):
        """
        :param hiddens: 隐层维度列表
        :param input_size: memory_size
        :param activation_function: 每一层都采用相同的激活函数？为啥不搞个激活函数列表？
        :param out_act: 是否对输出层加激活函数，False表示不加
        :param dropout_ratio: 不加dropout？是因为效果不好吗？
        """
        super(MLP, self).__init__()
        # dropout_ratio = 0.2
        # layers = [nn.Dropout(dropout_ratio)]
        layers = []  # 包含线性层和相应的激活函数

        previous_h = input_size
        for i, h in enumerate(hiddens):
            # out_act为false的时候，输出层不加激活
            activation = None if i == len(hiddens) - 1 and not out_act else activation_function
            layers.append(nn.Linear(previous_h, h))

            if activation is not None:
                layers.append(activation)

            # layers.append(nn.Dropout(dropout_ratio))
            previous_h = h
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class nconv(nn.Module):
    """
    张量运算
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl, vw->ncwl', (x, A))
        return x.contiguous()


######################################################################
# RNN layer
######################################################################
class GRU(nn.Module):
    def __init__(self, c_in, c_out):
        super(GRU, self).__init__()
        self.gru = nn.GRU(c_in, c_out, batch_first=True)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.gru(x)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class LSTM(nn.Module):
    def __init__(self, c_in, c_out,):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(c_in, c_out, batch_first=True)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.lstm(x)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# building blocks for Modeling temporal dependency
######################################################################
class CausalConv2d(nn.Conv2d):
    """
    单向padding，causal体现在kernel_size=2
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(0, self._padding),
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """
    dilated causal convolution layer with GLU function
    暂时用GTU代替
    """

    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        super(DCCLayer, self).__init__()
        # self.relu = nn.ReLU()
        # padding=0, 所以feature map尺寸会减小，最终要减小到1才行？如果没有，就要pooling
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation)
        self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation)
        # self.bn = nn.BatchNorm2d(c_out)
        # self.filter_conv = nn.Conv2d(c_in, c_out, kernel_size, dilation=dilation)
        # self.gate_conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        # x = self.relu(x)
        filter = (self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        output = filter * gate
        # output = self.bn(output)

        return output


class FF(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0.):
        super(FF, self).__init__()

        self.linear_1 = linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        # x = F.relu(self.linear_1(x))
        output = self.linear_2(x)

        return output


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, d_model, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.size = d_model

        self.gamma = nn.Parameter(torch.ones(self.size))
        self.beta = nn.Parameter(torch.zeros(self.size))

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return:
        """
        normalized = (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.epsilon)
        output = self.gamma * normalized + self.beta

        return output


def attention(q, k, v, d_k, mask=None):
    """
    :param q: [b, heads, T, d_k]
    :param mask:
    :return:
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # if mask is not None:  # 单向编码需要mask，取下对角阵
    #     mask = mask.unsqueeze(1)
    #     scores = scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)

    return output


class SepConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, ):
        super(SepConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: [b, C, T]
        :return:
        """
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0., kernel_size=3):
        """
        :param d_model: f_in
        :param heads:
        :param dropout:
        """
        super(ConvAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  # narrow multi-head
        self.h = heads
        pad = (kernel_size - 1) // 2
        self.to_q = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        self.to_k = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        self.to_v = SepConv1d(d_model, d_model, kernel_size, padding=pad)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout))

        self.attn = ProbAttention()

    def forward(self, x, attn_mask=None):
        """
        :param x: [b, T, d_model]
        :param attn_mask:
        :return: [b, T, d_model]
        """
        b = x.size(0)  # [b*N, T, C]
        x = x.transpose(1, 2)  # [b*N, C, T]

        # perform linear operation and split into N heads
        q = self.to_q(x).transpose(1, 2).contiguous().view(b, -1, self.h, self.d_k)  # [b*N, T, h, d_k]
        k = self.to_k(x).transpose(1, 2).contiguous().view(b, -1, self.h, self.d_k)
        v = self.to_v(x).transpose(1, 2).contiguous().view(b, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * heads * seq_len * d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = attention(q, k, v, self.d_k, attn_mask)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        output = self.to_out(concat)

        # scores, attn = self.attn(q, k, v)
        # concat = scores.view(b, -1, self.d_model)
        # output = self.to_out(concat)

        return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=12, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        # position = torch.tensor(torch.arange(seq_len), dtype=torch.long).to(DEVICE)
        # pe = self.pe(position).unsqueeze(0)
        x = x + pe
        return self.dropout(x)


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff=4, dropout=0.):
        super(Feedforward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return: [bs, T, d_model]
        """
        return self.net(x)


class TransformerLayer(nn.Module):
    """
    transformer layer with 4 heads
    """

    def __init__(self, d_model, heads=4, dropout=0., max_seq_len=12):
        # head=1时，ProbAttention内存占用爆炸，为什么？
        super(TransformerLayer, self).__init__()

        self.attn = ConvAttention(d_model, heads, dropout)
        self.ff = Feedforward(d_model, dropout=dropout)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe = PositionalEncoder(d_model)
        # self.pe = PositionalEmbedding(d_model)

    def forward(self, x):
        """
        输入多了207这一维，需要reshape为64*207，最后再reshape回来
        对不同的sample和不同的node建立共同的模型
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_in, N, T]
        """
        # b = x.size(0)
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]

        # x = self.pe(x)
        x2 = self.norm_1(x)
        x = x + self.dropout1(self.attn(x2))
        x2 = self.norm_2(x)
        x = x + self.dropout2(self.ff(x2))

        # x2 = self.attn(x)
        # x = x + self.dropout1(x2)
        # x = self.norm_1(x)
        # x2 = self.ff(x)
        # x = x + self.dropout2(x2)
        # x = self.norm_2(x)

        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        # new_x = self.attn(x)
        # x = x + self.dropout(new_x)
        # y = x = self.norm1(x)
        # y = (F.relu(self.conv1(y.transpose(-1, 1))))
        # y = (self.conv2(y).transpose(-1, 1))
        # x = self.norm2(x+y)

        x = x.reshape(b, -1, T, C)
        x = x.permute(0, 3, 1, 2)

        return x


######################################################################
# Informer encoder layer
######################################################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(DEVICE)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# class Trans(nn.Module):
#     def __init__(self, d_model, n_heads=4):
#         super(Trans, self).__init__()
#         self.trans = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=32)
#     def forward(self, x):
#         b, C, N, T = x.shape
#         x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
#         x = x.reshape(-1, T, C)  # [64*207, 12, 32]
#         x = x.transpose(0, 1)  # [12, 64*207, 32]
#         x = self.trans(x)
#         x = x.transpose(0, 1)
#         output = x.reshape(b, -1, T, C)
#         output = output.permute(0, 3, 1, 2)
#
#         return output


class InformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model


    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        # t1 = time.time()
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # print(f'full att time: {time.time() - t1}')
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # output = self.norm2(x + y)

        output = x.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# building blocks for Modeling spatial dependency
######################################################################

class GCNLayer(nn.Module):  # 所有building block的初始化参数和输入输出都应该统一吗？或者在所有building block上面封装一层？
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        """
        :param K: K-order
        :param cheb_polynomials: laplacian matrix？
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(GCNLayer, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k])
                T_k = alpha * self.cheb_polynomials[k] + beta * adp

                # T_k = self.cheb_polynomials[k]  # [N, N]
                self.theta_k = self.Theta[k]  # [c_in, c_out]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)
                # print(f'x: {graph_signal[0, 0, :]}')
                # print(f'output: {output[0, 0, :]}')

            outputs.append(output.unsqueeze(-1))

        outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        # print(f'outputs: {outputs[0, :, 0, :]}')
        return outputs


class DiffusionConvLayer(nn.Module):
    """
    K-order diffusion convolution layer
    """

    def __init__(self, K, supports, c_in, c_out, nodevec1, nodevec2):
        """

        :param K:
        :param supports: adj
        :param c_in:
        :param c_out:
        """
        super(DiffusionConvLayer, self).__init__()
        self.nconv = nconv()
        # c_in = (K * len(supports) + 1) * c_in
        c_in = (K * (len(supports) + 1) + 1) * c_in  # 这里supports应该+1吧？
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp = linear(c_in, c_out).to(DEVICE)
        self.K = K
        self.supports = supports
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        # x = self.relu(x)
        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug???
        # 差别不大
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        new_supports = self.supports + [adp]

        out = [x]  # 对应k=0
        for a in new_supports:
            # x.shape [b, dim, N, seq_len]
            # a.shape [b, N, N]
            x1 = self.nconv(x, a)  # 对应k=1
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)  # 对应k=2
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
        # h = self.bn(h)
        h = F.dropout(h, 0.3, training=self.training)  # 在pems数据集上不能加？

        return h

# class DiffusionConvLayer(nn.Module):
#     """
#     K-order diffusion convolution layer
#     """
#     def __init__(self, K, c_in, c_out, nodevec1, nodevec2):
#         """
#         :param K:
#         :param c_in:
#         :param c_out:
#         """
#         super(DiffusionConvLayer, self).__init__()
#         self.nconv = nconv()
#         c_in = (K + 1) * c_in
#         self.nodevec1 = nodevec1
#         self.nodevec2 = nodevec2
#         self.mlp = linear(c_in, c_out).to(DEVICE)
#         self.K = K
#         # self.relu = nn.ReLU()
#         # self.bn = nn.BatchNorm2d(c_out)
#
#     def forward(self, x):
#         # x = self.relu(x)
#         # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug???
#         # 差别不大
#         adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
#         mask = torch.zeros_like(adp) - 10 ** 10
#         adp = torch.where(adp == 0, mask, adp)
#         adp = F.softmax(adp, dim=1)
#         new_supports = [adp]
#
#         out = [x]  # 对应k=0
#         for a in new_supports:
#             # x.shape [b, dim, N, seq_len]
#             # a.shape [b, N, N]
#             x1 = self.nconv(x, a)  # 对应k=1
#             out.append(x1)
#             for k in range(2, self.K + 1):
#                 x2 = self.nconv(x1, a)  # 对应k=2
#                 out.append(x2)
#                 x1 = x2
#
#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
#         # h = self.bn(h)
#         # h = F.dropout(h, 0.3, training=self.training)  # 在pems数据集上不能加？
#
#         return h


# class Diff(nn.Module):
#     def __init__(self, K, supports, c_in, c_out, nodevec1, nodevec2):
#         """
#
#         :param K:
#         :param supports: adj
#         :param c_in:
#         :param c_out:
#         """
#         super(Diff, self).__init__()
#         self.nconv = nconv()
#         c_in = (K * (len(supports) + 1) + 1) * c_in
#         self.nodevec1 = nodevec1
#         self.nodevec2 = nodevec2
#         self.mlp = linear(c_in, c_out)
#         self.K = K
#         self.supports = supports
#
#     def forward(self, x):
#         """
#         :param x: [b, C, N, T]
#         :return:
#         """
#         # x = self.relu(x)
#         # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug???
#         # 差别不大
#         adp = F.relu(torch.matmul(self.nodevec1, self.nodevec2))
#         mask = torch.zeros_like(adp) - 10 ** 10
#         adp = torch.where(adp == 0, mask, adp)
#         adp = F.softmax(adp, dim=-1)  # [T, N, N]
#
#         out = [x]  # 对应k=0
#         for a in self.supports:
#             x1 = self.nconv(x, a)  # 对应k=1
#             out.append(x1)
#             for k in range(2, self.K + 1):
#                 x2 = self.nconv(x1, a)  # 对应k=2
#                 out.append(x2)
#                 x1 = x2
#
#         # ncvl, vw->ncwl
#         # (b, C, N, T)(N, N)->(b, C, N, T)
#         # (b, C, N, T)(T, N, N)->()
#         # (b, T, C, N)(T, N, N)->()
#         x0 = torch.clone(x)
#         x0 = x0.permute(0, 3, 1, 2)  # (b, T, C, N)
#         x1 = torch.matmul(x0, adp)
#         out.append(x1.permute(0, 2, 3, 1))
#         for k in range(2, self.K + 1):
#             x2 = torch.matmul(x1, adp)
#             out.append(x2.permute(0, 2, 3, 1))
#             x1 = x2
#
#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
#         # h = self.bn(h)
#         # h = F.dropout(h, 0.3, training=self.training)  # necessary?
#
#         return h
#
#
# class Diff2(nn.Module):
#     def __init__(self, K, supports, c_in, c_out, nodevec1, nodevec2, alpha):
#         """
#
#         :param K:
#         :param supports: adj
#         :param c_in:
#         :param c_out:
#         """
#         super(Diff2, self).__init__()
#         self.nconv = nconv()
#         c_in = (K * len(supports) + 1) * c_in
#         self.nodevec1 = nodevec1
#         self.nodevec2 = nodevec2
#         self.mlp = linear(c_in, c_out)
#         self.K = K
#         self.supports = [s.unsqueeze(0).repeat(12, 1, 1) for s in supports]
#         self.alpha = alpha
#
#     def forward(self, x):
#         """
#         :param x: [b, C, N, T]
#         :return:
#         """
#         adp = F.relu(torch.matmul(self.nodevec1, self.nodevec2))
#         mask = torch.zeros_like(adp) - 10 ** 10
#         adp = torch.where(adp == 0, mask, adp)
#         adp = F.softmax(adp, dim=-1)  # [T, N, N]
#
#         # for i in range(len(self.supports)):  # 反思一下这个bug
#         #     self.supports[i] = torch.FloatTensor([0.8]) * self.supports[i] + torch.FloatTensor([0.2]) * adp
#
#         out = [x]
#         x = x.permute(0, 3, 1, 2)  # (b, T, C, N)
#         for i in range(len(self.supports)):
#             alpha, beta = F.softmax(self.alpha[i])
#             a = alpha * self.supports[i] + beta * adp
#             x1 = torch.matmul(x, a)  # (b, T, C, N)(T, N, N)->
#             out.append(x1.permute(0, 2, 3, 1))
#             for k in range(2, self.K + 1):
#                 x2 = torch.matmul(x1, a)
#                 out.append(x2.permute(0, 2, 3, 1))
#                 x1 = x2
#
#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
#         # h = self.bn(h)
#         # h = F.dropout(h, 0.3, training=self.training)  # necessary?
#
#         return h


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, A_wave, c_in, c_out, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = c_in
        self.out_features = c_out
        A_wave = torch.from_numpy(A_wave).to(DEVICE)
        self.A_wave = A_wave
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        :param input: [batch_size, f_in, N, T]
        :param A_wave: Normalized adjacency matrix.
        :return:
        """
        x = input.permute(0, 2, 3, 1)  # [B, N, T, F]
        # x = self.relu(x)
        lfs = torch.einsum("ij,jklm->kilm", [self.A_wave, x.permute(1, 0, 2, 3)])
        output = F.relu(torch.matmul(lfs, self.weight))  # relu先不要吧？
        # output = (torch.matmul(lfs, self.weight))

        if self.bias is not None:
            output = output + self.bias

        output = output.permute(0, 3, 1, 2)
        # output = self.bn(output)

        return output


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


class SpatialInformer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialInformer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialProbAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)  # spatial transformer需要pe吗？
        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output


class SpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialFullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 在这里加上fixed邻接矩阵？
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SpatialProbAttention(nn.Module):
    def __init__(self, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)  # # [256*12, 4, 8]
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [256*12, 4, 207, 8]
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) [256*12, 4, 18, 207]

        # print(context_in.shape)  # [256*12, 4, 207, 8]
        # print(torch.matmul(attn, V).shape)  # [256*12, 4, 18, 8]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 部分赋值
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print(index.shape)  # [256*12, 4, 18]

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # [256*12, 4, 18, 207] 18=sqrt(207)*3
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn
