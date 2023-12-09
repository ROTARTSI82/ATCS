import numpy as np
import torch
import random
import math
import struct
import json

enc_dict = {i.to_bytes(1, 'big'): i for i in range(256)}
tokens = [i.to_bytes(1, 'big') for i in range(256)]
with open("vocab.txt", "rb") as fp:
    vocab = fp.read()

d_model = 512
vocab_size = 1279  # 767
ctx = 512
nhead = 8
nlayer = 4
lab_smooth = 0.1

longest = 0
idx = 256
for i in range(vocab_size - 256):
    s = struct.unpack("@L", vocab[:8])[0]
    utf = vocab[8:8+s]
    if utf == b"newdoc":
        print(i, idx, utf)
    longest = max(longest, len(utf))
    enc_dict[utf] = idx
    tokens.append(utf)
    vocab = vocab[8+s:]
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


# with open("MASTER.txt", "rb") as fp:
#     master = fp.read()
#
# data = tokenize(master)
# with open('dataset_enc_master.json', 'w') as fp:
#     json.dump(data, fp)

# with open('dataset_enc_master.json', 'r') as fp:
#    data = json.load(fp)

with open("dataset.txt", 'rb') as fp:
    data = fp.read()

npdtype = float
dtype = torch.float32
device = torch.device('cuda')

pos_enc = [[(math.sin if i%2 == 0 else math.cos)(pos / 10000**(2*(i//2) / d_model)) for i in range(d_model)]
           for pos in range(ctx)]
pos_enc = torch.tensor(pos_enc, dtype=dtype, device=device)


class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embed = torch.nn.Embedding(vocab_size, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, activation=torch.nn.ReLU(),
                                                     batch_first=True, norm_first=True, bias=True, dropout=0.1)
        self.trans = torch.nn.TransformerEncoder(enc_layer,
                                                 nlayer, torch.nn.LayerNorm(d_model, bias=True))
        # todo: come up with a better way of doing prob_out
        self.prob_out = torch.nn.Sequential(torch.nn.Linear(d_model, vocab_size, bias=False))

    def forward(self, txt, ctxlen):
        # print(txt.shape)
        # print(self.vocab_embed(txt).shape, pos_enc[:ctxlen].shape)
        t_in = self.vocab_embed(txt) + pos_enc[:ctxlen]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(ctxlen, device=device, dtype=dtype)
        trans = self.trans(t_in, mask=mask, is_causal=True)
        return self.prob_out(trans)


lr = 1.5e-4
model = LanguageModel().to(device).train(True)
opt = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=lab_smooth)

print(f"{sum(tens.numel() for tens in model.parameters())} parameters")

load = torch.load('discordllm_BIG.txt')
model.load_state_dict(load['model'], strict=False)

start = 0
start = load['start']

tot_steps = 1
warmup = 100
running = True
while running:
    opt.zero_grad()

    # lr = d_model**-0.5 * min(tot_steps**-0.5, tot_steps*warmup**-1.5)
    # for g in opt.param_groups:
    #     g['lr'] = lr
    vec = []
    for minibatch in range(16):
        toks = tokenize(data[start:start+4*4096])
        vec.append(toks[:ctx + 1])
        # start += random.randint(-16, 1024 // 12 - 16)
        start += random.randint(-ctx//2, 2*ctx)
        start = max(0, start)
        if start >= len(data) - 4*4096:
            start %= len(data) - 4*4096
            running = False
    # start -= 8  # random.randint(0, 8)

    case = torch.tensor(vec, device=device)
    targ = torch.nn.functional.one_hot(case[:, 1:], num_classes=vocab_size).float()
    out = model(case[:, :ctx], ctx)

    loss = loss_fn(out, targ)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    print(f"{start}\t{100*start/len(data):.2f}% {lr:.2E}\t{loss.item()}\t{b''.join(tokens[i] for i in data[start:start+16])}")

    if tot_steps % 64 == 0 or not running:
        print("============ saved checkpoint =================")
        torch.save({'model': model.state_dict(), 'start': start}, 'discordllm_BIG.txt' if running else "FINAL.txt")
    tot_steps += 1

generate = True
if generate:
    prompt = tokenize(b" Capture of Sedalia\n The capture of Sedalia occurred during the American Civil War when a Confederate force captured the Union garrison of Sedalia, Missouri, on October 15, 1864.")
    print(prompt)

    case = torch.tensor(prompt[-ctx:], device=device)
    out = torch.softmax(model(case, min(ctx, len(prompt))), dim=1)
    print(b''.join(tokens[i] for i in torch.argmax(out, dim=1)).decode('utf-8', 'ignore'))

    model.train()
    for i in range(1024):
        case = torch.tensor(prompt[-ctx:], device=device)
        out = torch.softmax(model(case, min(ctx, len(prompt))), dim=1)
        # new = torch.multinomial(out[-1], num_samples=1)[0]
        new = torch.argmax(out[-1])
        prompt.append(new.item())
        print(i, tokens[new.item()])
    print('\n')
    print(b''.join(tokens[i] for i in prompt).decode('utf-8', 'ignore'))
