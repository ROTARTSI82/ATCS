import torch

dtype = torch.float32
device = torch.device('cuda')

d_model = 512
n_head = 8
d_hid = 2048
n_enc = 6
n_dec = 6
qk_size = d_model // n_head
v_size = d_model // n_head
drop = 0


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = torch.nn.LayerNorm(d_model)
        self.kv_proj = torch.nn.Linear(d_model, (qk_size + v_size) * n_head, bias=False)
        self.q_proj = torch.nn.Linear(d_model, qk_size * n_head, bias=False)
        self.out_proj = torch.nn.Linear(v_size * n_head, d_model, bias=False)
        self.final_dropout = torch.nn.Dropout(drop)

    def forward(self, x, attn):
        """
        x is a (batch, tokens, channels) tensor
        """
        B, T, C = x.size()
        BA, TA, CA = attn.size()

        # BA should equal B and C == CA
        # print(B, BA, C, CA)
        assert B == BA and C == CA

        q = self.q_proj(self.q_norm(x)).split([qk_size * n_head], dim=2)[0]
        k, v = self.kv_proj(attn).split([qk_size * n_head, v_size * n_head], dim=2)
        q = q.view(B, T, n_head, qk_size).transpose(1, 2)  # B x n_heads x T x qk_size
        k = k.view(BA, TA, n_head, qk_size).transpose(1, 2)  # BA x n_heads x TA x qk_size
        v = v.view(BA, TA, n_head, v_size).transpose(1, 2)  # BA x n_heads x TA x v_size

        """
        kT = k.transpose(2, 3)  # B x n_heads x qk_size x T
        correlations = self.softmax((q @ kT) / dim_scale) # B x n_heads x T x T
        # encoder: no masking of the correlations happens here
        o = correlations @ v # B x n_heads x T x v_size
        """
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=drop)

        o_shaped = o.transpose(1, 2).reshape(B, T, n_head * v_size)
        return self.final_dropout(self.out_proj(o_shaped))  # B x T x C


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_lin = torch.nn.LayerNorm(d_model)

        self.lin_swish = torch.nn.Linear(d_model, d_hid, bias=False)
        self.lin_gate = torch.nn.Linear(d_model, d_hid, bias=False)
        self.lin_out = torch.nn.Linear(d_hid, d_model, bias=False)
        self.mlp_drop = torch.nn.Dropout(drop)

    def forward(self, x):
        norm = self.norm_lin(x)
        swish = self.lin_swish(norm)
        gated = swish * torch.nn.functional.sigmoid(swish) * self.lin_gate(norm)
        x = x + self.mlp_drop(self.lin_out(gated))
        return x


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_enc = torch.nn.Embedding(2, d_model)
        self.to_dec = torch.nn.Parameter(torch.randn(3, d_model))
        self.encoder = Attention()
        self.enc_mlp = MLP()
        self.dec = Attention()
        self.cross = Attention()
        self.dec_mlp = MLP()
        self.enc_norm = torch.nn.LayerNorm(d_model)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.outp = torch.nn.Linear(d_model, 2)

    def forward(self, inp):
        enc = self.to_enc(inp)
        enc = enc + self.encoder(enc, enc)
        enc = self.enc_norm(enc + self.enc_mlp(enc))

        dec = self.to_dec.repeat(4, 1, 1)
        dec = dec + self.dec(dec, dec)
        dec = dec + self.cross(dec, enc)
        dec = dec + self.dec_mlp(dec)

        return self.outp(self.final_norm(dec))


if __name__ == "__main__":
    model = Transformer().to(device).train()
    opt = torch.optim.AdamW(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)

    params = sum(i.numel() for i in model.parameters())
    print("Training transformer with", params, "parameters\n")

    inps = []
    outs = []
    for a in range(2):
        for b in range(2):
            inps.append([a, b])
            outs.append([a ^ b, a & b, a | b])

    inps = torch.tensor(inps, device=device)
    outs = torch.tensor(outs, device=device)

    for epoch in range(1024):
        opt.zero_grad()
        model_out = model(inps)
        # print(model_out.shape, model_out)
        # print(outs.shape, outs)
        loss = loss_fn(model_out.view(-1, 2), outs.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

        z = zip(inps.tolist(), torch.argmax(model_out, dim=2).tolist())
        print([(i, v) for i, v in z], loss.item())

    print([i for i in model.parameters()])
