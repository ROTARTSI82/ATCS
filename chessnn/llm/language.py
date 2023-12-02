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

longest = 0
idx = 256
for i in range(512):
    s = struct.unpack("@L", vocab[:8])[0]
    utf = vocab[8:8+s]
    if utf == b"newdoc":
        print(i, idx, utf)
    longest = max(longest, len(utf))
    enc_dict[utf] = idx
    tokens.append(utf)
    vocab = vocab[8+s:]
    idx += 1


# with open("dataset.txt", "rb") as fp:
#   data = fp.read()

def tokenize(string):
    dat_encoded = []
    enc_str = []
    i = 0
    while i < len(string):
        if i % 4096 == 0:
            print(f"{100*i/len(string):.2f}% {string[i:i+16]}")
        for j in reversed(range(1, longest + 1)):
            sub = string[i:i+j]
            if sub in enc_dict:
                dat_encoded.append(enc_dict[sub])
                enc_str.append(sub)
                i += len(sub)
                break
        else:
            print(b"what?: " + string[i:i+16])
            print(string[i-16:i+16])
            dat_encoded.append(int.from_bytes(string[i], 'big'))
            enc_str.append(string[i])
            i += 1
    return dat_encoded


# with open('dataset_enc.json', 'w') as fp:
#    json.dump(dat_encoded, fp)
# print(enc_str)


with open('dataset_enc.json', 'r') as fp:
    data = json.load(fp)

npdtype = float
dtype = torch.float32
device = torch.device('cuda')

d_model = 512 + 256
vocab_size = 512 + 256

pos_enc = [[(math.sin if i%2 == 0 else math.cos)(pos / 10000**(2*(i//2) / d_model)) for i in range(d_model)]
           for pos in range(1024)]
pos_enc = torch.tensor(pos_enc, dtype=dtype, device=device)


class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embed = torch.nn.Embedding(vocab_size, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model, 8, activation=torch.nn.LeakyReLU(0.01),
                                                     batch_first=True, norm_first=True, bias=True, dropout=0.1)
        self.trans = torch.nn.TransformerEncoder(enc_layer,
                                                 6, torch.nn.LayerNorm(d_model, bias=True))
        # todo: come up with a better way of doing prob_out
        self.prob_out = torch.nn.Sequential(torch.nn.Linear(d_model, vocab_size, bias=False))

    def forward(self, txt, ctxlen):
        # print(txt.shape)
        # print(self.vocab_embed(txt).shape, pos_enc[:ctxlen].shape)
        t_in = self.vocab_embed(txt) + pos_enc[:ctxlen]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(ctxlen, device=device, dtype=dtype)
        trans = self.trans(t_in, mask=mask, is_causal=True)
        return self.prob_out(trans)


lr = 3e-4
model = LanguageModel().to(device).train(True)
opt = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

load = torch.load('discordllm_BIG.txt')
model.load_state_dict(load['model'])

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
    for minibatch in range(4):
        vec.append(data[start:start + 1025])
        start += random.randint(1, 8)
        if start >= len(data) - 1025:
            start = 0
            running = False
    # start -= 8  # random.randint(0, 8)

    case = torch.tensor(vec, device=device)
    targ = torch.nn.functional.one_hot(case[:, 1:], num_classes=vocab_size).float()
    out = model(case[:, :1024], 1024)

    loss = loss_fn(out, targ)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    print(f"{start}\t{100*start/len(data):.2f}% {lr:.2E}\t{loss.item()}\t{b''.join(tokens[i] for i in data[start:start+16])}")

    if tot_steps % 64 == 0 or not running:
        print("============ saved checkpoint =================")
        torch.save({'model': model.state_dict(), 'start': start}, 'discordllm_BIG.txt' if running else "FINAL.txt")
    tot_steps += 1

generate = True
if generate:
    prompt = tokenize(b"To be the very best, one must")
    print(prompt)

    for i in range(1000):
        case = torch.tensor(prompt[-1024:], device=device)
        out = model(case, min(1024, len(prompt)))
        new = torch.argmax(out[-1])
        prompt.append(new.item())
        print(i, tokens[new.item()])
    print(b''.join(tokens[i] for i in prompt))