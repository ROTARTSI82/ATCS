import math

import torch
import torch.nn as nn

dev = torch.device("cpu")

activ = lambda: torch.nn.LeakyReLU(0.01)

qk_size = 4
v_size = 4
n_heads = 8
channels = 32
nn_dim = 128

n_blocks = 3

inp_size = 1
out_size = 1

dropout = 0

dim_scale = math.sqrt(qk_size)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(channels, (qk_size * 2 + v_size) * n_heads, bias=False)
        self.out_proj = nn.Linear(v_size * n_heads, channels, bias=False)
        self.final_dropout = nn.Dropout(dropout) 
        self.softmax = nn.Softmax(dim=-1) # softmax along matrix ROWS

    def forward(self, x):
        """
        x is a (batch, tokens, channels) tensor
        """
        B, T, C = x.size()

        q, k, v = self.qkv_proj(x).split([qk_size * n_heads, qk_size * n_heads, v_size * n_heads], dim=2)
        q = q.view(B, T, n_heads, qk_size).transpose(1, 2)  # B x n_heads x T x qk_size
        k = k.view(B, T, n_heads, qk_size).transpose(1, 2)
        v = v.view(B, T, n_heads, v_size).transpose(1, 2)  # B x n_heads x T x v_size


        """
        kT = k.transpose(2, 3)  # B x n_heads x qk_size x T
        correlations = self.softmax((q @ kT) / dim_scale) # B x n_heads x T x T
        # encoder: no masking of the correlations happens here
        o = correlations @ v # B x n_heads x T x v_size
        """
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

        o_shaped = o.transpose(1, 2).reshape(B, T, n_heads * v_size)
        return self.final_dropout(self.out_proj(o_shaped))  # B x T x C


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.norm_lin = nn.LayerNorm(channels)

        self.attn = Attention()
        self.mlp = nn.Sequential(nn.Linear(channels, nn_dim), activ(), nn.Linear(nn_dim, channels))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_lin(x))
        return self.drop(x)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(inp_size, channels), activ(), nn.Dropout(dropout),
                                 *[Block() for _ in range(n_blocks)], nn.Linear(channels, out_size))

    def forward(self, x):
        return self.seq(x)


model = Transformer().to(dev).train()
opt = torch.optim.Adam(model.parameters())

loss_fn = torch.nn.MSELoss()

inps = torch.tensor([[[0], [1]], [[1], [0]], [[0], [0]], [[1], [1]]], dtype=torch.float32).to(dev)
outs = torch.tensor([[[1], [1]], [[1], [1]], [[0], [0]], [[0], [0]]], dtype=torch.float32).to(dev)

epoch = 1
while True:
    opt.zero_grad()
    model_out = model(inps)
    loss = loss_fn(model_out, outs)
    loss.backward()
    opt.step()

    epoch += 1
    if epoch % 1024 == 0:
        print(inps, " -> ", model_out, "\t", loss.item())
