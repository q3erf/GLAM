import torch
import torch.nn as nn
from module.sinkhorn import Sinkhorn

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(CrossAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        # d_k^{-0.5}
        self.scale = hidden_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.sinkhorn = Sinkhorn(5)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.sigmoid(x) # [b, h, q_len, k_len]

        m = torch.stack([self.sinkhorn(x[:,i,:,:])  for i in range(x.size(1))], dim=1) # [b, h, q_len, k_len]

        y = self.att_dropout(m)
        y = y.matmul(v)  # [b, h, q_len, attn]

        y = y.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        y = y.view(batch_size, -1, self.head_size * d_v)

        y = self.output_layer(y)
        assert y.size() == orig_q_size
        return m, y