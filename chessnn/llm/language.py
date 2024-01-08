import numpy as np
from torchviz import make_dot
import torch
import random
import math
import struct
import json
from custom_transform import *

enc_dict = {i.to_bytes(1, 'big'): i for i in range(256)}
tokens = [(i.to_bytes(1, 'big'), -1, -1) for i in range(256)]
with open("vocab.bin", "rb") as fp:
    vocab = fp.read()

vocab_size = 3864  # 767
ctx = 512
lab_smooth = 0.1

temp = 0.8
top_k = int(128)

model_file = "bigger_boi_weight_tie.bin.v2"
conf = Config()
mats = make_rot_mats(conf.d_model, ctx)

longest = 1
idx = 256
for i in range(vocab_size - 256):
    l = struct.unpack("@I", vocab[:4])[0]
    r = struct.unpack("@I", vocab[4:8])[0]
    s = struct.unpack("@I", vocab[8:12])[0]
    
    utf = vocab[12:12+s]
    if utf == b"newdoc":
        print(i, idx, utf)
    longest = max(longest, len(utf))
    enc_dict[utf] = idx
    tokens.append((utf, l, r))
    vocab = vocab[12+s:]
    idx += 1
longest = 24  # hardcoding this because


def tokenize(string):
    dat_encoded = []
    enc_str = []
    occ_table = dict()
    i = 0
    while i < len(string):
        if i % 4096 == 0:
            pass  # print(f"{100*i/len(string):.2f}% {string[i:i+16]}")
        for j in reversed(range(1, longest + 1)):
            sub = string[i:i+j]
            if sub in enc_dict:
                idx = enc_dict[sub]
                dat_encoded.append(idx)
                enc_str.append(sub)
                i += len(sub)

                if idx in occ_table:
                    occ_table[idx] += 1
                else:
                    occ_table[idx] = 1
                break
        else:
            print(b"what?: " + string[i:i+16])
            print(string[i-16:i+16])
            dat_encoded.append(int.from_bytes(string[i], 'big'))
            enc_str.append(string[i])
            i += 1
    # print(sorted([(i, occ_table[i]) for i in occ_table], key=lambda x: x[1]))
    return dat_encoded


def tokenize2(string):
    stream = [i for i in string]
    for idx, i in enumerate(tokens[256:]):
        cpy = []
        j = 0
        while j < len(stream):
            if j < len(stream) - 1 and stream[j] == i[1] and stream[j+1] == i[2]:
                cpy.append(idx + 256)
                j += 2
            else:
                cpy.append(stream[j])
                j += 1
        stream = cpy
    return stream


with open("dataset.txt", 'rb') as fp:
    data = fp.read()

npdtype = float
dtype = torch.float32


class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embed = torch.nn.Embedding(vocab_size, conf.d_model)
        self.encoder = torch.nn.ModuleList([Block(conf, causal=True) for _ in range(12)])
        self.fnorm = RMSNorm(conf.d_model)
        self.prob_out = torch.nn.Sequential(torch.nn.Linear(conf.d_model, vocab_size, bias=False))
        self.vocab_embed.weight = self.prob_out[0].weight

    def forward(self, txt, mats):
        # print(txt.shape)
        # print(self.vocab_embed(txt).shape, pos_enc[:ctxlen].shape)
        t_in = rope(self.vocab_embed(txt), mats)
        for layer in self.encoder:
            t_in = layer(t_in)
        return self.prob_out(self.fnorm(t_in))


lr = 3e-4
model_raw = LanguageModel().to(device).train(True)
opt = torch.optim.AdamW(model_raw.parameters(), lr=lr, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=lab_smooth)

print(f"{sum(tens.numel() for tens in model_raw.parameters()) / 1000000} million parameters")

load = torch.load(model_file, map_location=device)
model_raw.load_state_dict(load['model'])
# model = torch.jit.script(model_raw)
model = model_raw
print(model)

start = load['start']
start = len(data) - 1024*4096

tot_steps = 1
warmup = 100
losses = []
running = False
while running:
    opt.zero_grad()

    # lr = d_model**-0.5 * min(tot_steps**-0.5, tot_steps*warmup**-1.5)
    # for g in opt.param_groups:
    #     g['lr'] = lr
    vec = []
    for minibatch in range(12):
        rng = random.random() < 1/3
        cur = random.randint(0, len(data) - 16*ctx) if rng else start
        toks = tokenize(data[cur:cur+16*ctx])
        vec.append(toks[4:ctx + 5])

        if not rng:
            start += random.randint(ctx, 8*ctx)
            start = max(0, start)
            if start >= len(data) - 16*ctx:
                start %= len(data) - 16*ctx
                running = False

    case = torch.tensor(vec, device=device)
    out = model(case[:, :ctx])

    targ = case[:, 1:].reshape(-1)
    loss = loss_fn(out.view(-1, vocab_size), targ)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    print(f"{start}\t{100*start/len(data):.2f}% {lr:.2E}\t{loss.item()}\t{b''.join(tokens[i][0] for i in data[start:start+48])}")
    losses.append(loss.item())

    if tot_steps % 64 == 0 or not running:
        print("============ saved checkpoint =================")
        torch.save({'model': model.state_dict(), 'start': start}, model_file + ".v2")
        with open('loss.csv', 'a') as fp:
            fp.write("".join([", " + str(a) for a in losses]))
            losses = []
    tot_steps += 1


# torch.autograd.set_grad_enabled(False)
generate = True
txt = """
arma virumque cano, Troiae qui primus ab oris Italiam
""".encode('utf-8')

if generate:
    model.eval()
    prompt = tokenize(txt)
    print(prompt)

    print(txt.decode('utf-8'), end='')
    case = torch.tensor([prompt[-ctx:]], device=device)
    out = torch.softmax(model(case, mats)[0], dim=1)
    #print(b''.join(tokens[i][0] for i in torch.argmax(out, dim=1)).decode('utf-8', 'ignore'))

    for i in range(512):
        case = torch.tensor([prompt[-ctx:]], device=device)
        logits = model(case, mats)[0] / temp
        out = torch.softmax(logits, dim=1)

        if i == 300:
            make_dot(logits, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("attached", format="svg")

        # from nanogpt
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        new = torch.multinomial(out[-1], num_samples=1)[0]
        # new = torch.argmax(out[-1])
        prompt.append(new.item())
        print(tokens[new.item()][0].decode('utf-8', 'ignore'), end='', flush=True)
    # print(f'\n\n{"="*16}\n\n')
    # print(b''.join(tokens[i][0] for i in prompt).decode('utf-8', 'ignore'))


"""
messages = []
for a in range(vocab_size):
    va = model.vocab_embed(torch.tensor([a], device=device))[0]
    for b in range(a + 1, vocab_size):
        vb = model.vocab_embed(torch.tensor([b], device=device))[0]
        # print(va, vb)
        prod = torch.dot(va, vb)
        ma, mb = torch.linalg.norm(va), torch.linalg.norm(vb)
        csim = prod / (ma * mb)
        ang = torch.acos(csim)
        messages.append((csim.item(),
                         f"{tokens[a][0]} + {tokens[b][0]} -> {prod:.4} ({ma:.4}, {mb:.4})\t{csim:.3} {ang * 180 / torch.pi:.8}deg)"))
    print(f"{tokens[a][0]}\t{100*a/vocab_size}%", flush=True)

for i in sorted(messages, key=lambda x: x[0]):
    print(i[1], flush=True)
"""

# torch.jit.save(model, 'test.pt')
