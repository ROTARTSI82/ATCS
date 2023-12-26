import torch
from rmsnorm import RMSNorm
from math import sin, cos
from dataclasses import dataclass

dtype = torch.float32
device = torch.device('cuda')


@dataclass
class Config:
    d_model = 512
    n_head = 8
    d_hid = 512 * 8 // 3
    n_enc = 6
    n_dec = 2
    qk_size = 64
    v_size = 64
    drop = 0.1


# ctx by d_model//2 by 2 by 2
def make_rot_mats(d_model, ctx):
    thetas = [10000**(-2*i/d_model) for i in range(d_model//2)]
    arr = [[[[cos(m*t), -sin(m*t)], [sin(m*t), cos(m*t)]] for t in thetas] for m in range(ctx)]
    return torch.tensor(arr, device=device, dtype=dtype)


# batch is batch_size by ctx by d_model
def rope(batch, mats):
    B, T, C = batch.size()
    trans = torch.einsum("thxy,bthx->bthy", mats[:T], batch.reshape(B, T, C//2, 2))
    return trans.reshape(B, T, C)


class Attention(torch.nn.Module):
    def __init__(self, conf: Config, causal=False, do_q_proj=True, bias_out=False):
        super().__init__()
        self.conf = conf
        self.causal = causal
        self.kv_proj = torch.nn.Linear(conf.d_model, (conf.qk_size + conf.v_size) * conf.n_head, bias=False)
        self.q_proj = torch.nn.Linear(conf.d_model, conf.qk_size * conf.n_head, bias=False) if do_q_proj else None
        self.out_proj = torch.nn.Linear(conf.v_size * conf.n_head, conf.d_model, bias=bias_out)
        self.final_dropout = torch.nn.Dropout(conf.drop)

    def forward(self, x, attn):
        """
        x is a (batch, tokens, channels) tensor
        """
        B, T, C = x.size()
        BA, TA, CA = attn.size()

        # BA should equal B and C == CA
        # print(B, BA, C, CA)
        assert self.q_proj is None or B == BA and C == CA

        c = self.conf
        # ultra cursed as fuck!
        q = (self.q_proj(x) if self.q_proj is not None else x).split([c.qk_size * c.n_head], dim=2)[0]
        k, v = self.kv_proj(attn).split([c.qk_size * c.n_head, c.v_size * c.n_head], dim=2)
        q = q.view(B, T, c.n_head, c.qk_size).transpose(1, 2)  # B x n_heads x T x qk_size
        k = k.view(BA, TA, c.n_head, c.qk_size).transpose(1, 2)  # BA x n_heads x TA x qk_size
        v = v.view(BA, TA, c.n_head, c.v_size).transpose(1, 2)  # BA x n_heads x TA x v_size

        """
        kT = k.transpose(2, 3)  # B x n_heads x qk_size x T
        correlations = self.softmax((q @ kT) / dim_scale) # B x n_heads x T x T
        # encoder: no masking of the correlations happens here
        o = correlations @ v # B x n_heads x T x v_size
        """
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=device) if self.causal else None
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=c.drop,
                                                             is_causal=self.causal, attn_mask=mask)

        o_shaped = o.transpose(1, 2).reshape(B, T, c.n_head * c.v_size)
        return self.final_dropout(self.out_proj(o_shaped))  # B x T x C


class MLP(torch.nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.norm_lin = RMSNorm(conf.d_model)

        self.lin_swish = torch.nn.Linear(conf.d_model, conf.d_hid, bias=False)
        self.lin_gate = torch.nn.Linear(conf.d_model, conf.d_hid, bias=False)
        self.lin_out = torch.nn.Linear(conf.d_hid, conf.d_model, bias=False)
        self.mlp_drop = torch.nn.Dropout(conf.drop)

    def forward(self, x):
        norm = self.norm_lin(x)
        gated = torch.nn.functional.silu(self.lin_swish(norm)) * self.lin_gate(norm)
        x = x + self.mlp_drop(self.lin_out(gated))
        return x


class Block(torch.nn.Module):
    def __init__(self, conf: Config, cross=False, mlp=True, cross_first=False, causal=False):
        super().__init__()
        self.attn1 = Attention(conf, causal)
        self.attn2 = Attention(conf, causal) if cross else None
        self.mlp = MLP(conf) if mlp else None
        self.norm_a1 = RMSNorm(conf.d_model)
        self.norm_a2 = RMSNorm(conf.d_model) if cross else None
        self.cross_first = cross_first

    def do_cross(self, x, cross):
        norm_x2 = self.norm_a2(x)
        x = x + self.attn2(norm_x2, cross)
        return x

    def forward(self, x, cross=None):
        if cross is not None and self.cross_first:
            x = self.do_cross(x, cross)
        if self.attn1 is not None:
            norm_x = self.norm_a1(x)
            x = x + self.attn1(norm_x, norm_x)

        if cross is not None and not self.cross_first:
            x = self.do_cross(x, cross)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, conf: Config, cross_first=True):
        super().__init__()
        self.encoder = torch.nn.ModuleList([Block(conf) for _ in range(conf.n_enc)])
        self.decoder = torch.nn.ModuleList([
            Block(conf, cross=True, cross_first=cross_first) for _ in range(conf.n_dec)
        ])
        self.post_norm_enc = RMSNorm(conf.d_model)

    def forward(self, e, d):
        for layer in self.encoder:
            e = layer(e)
        e = self.post_norm_enc(e)

        for layer in self.decoder:
            d = layer(d, e)
        return d, e


if __name__ == "__main__":
    def test_custom_transform():
        c = Config()
        model = Transformer(c).to(device).train()
        emb = torch.nn.Embedding(2, c.d_model, device=device)
        oproj = torch.nn.Linear(c.d_model, 2, device=device, bias=False)
        to_dec = torch.randn(3, c.d_model).to(device)

        opt = torch.optim.AdamW([*model.parameters(), emb.weight, oproj.weight, to_dec], lr=3e-4, eps=1e-5,
                                betas=(0.9, 0.95), weight_decay=0.1)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        params = sum(i.numel() for i in model.parameters())
        print("Training transformer with", params / 1000000, " million parameters\n")

        inps = []
        outs = []
        for a in range(2):
            for b in range(2):
                inps.append([a, b])
                outs.append([a ^ b, a & b, a | b])

        inps = torch.tensor(inps, device=device)
        outs = torch.tensor(outs, device=device)

        for epoch in range(128):
            opt.zero_grad()
            model_out = oproj(model(emb(inps), to_dec.repeat(4, 1, 1))[0])
            # print(model_out.shape, model_out)
            # print(outs.shape, outs)
            loss = loss_fn(model_out.view(-1, 2), outs.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()

            z = zip(inps.tolist(), torch.argmax(model_out, dim=2).tolist())
            print([(i, v) for i, v in z], loss.item())

        print([i for i in model.parameters()])
    test_custom_transform()
