from scipy.io.wavfile import read, write
import scipy.signal as signal
import numpy as np
import torch
import random
import math

npdtype = float
dtype = torch.float32
device = torch.device('cuda')
chunk = 1025

FLEN = 511
OVERLAP = None
WIND = "hann"

SR = 16000

# print(stft[1024], data[1024])
# stft += 16 * np.random.randn(*stft.shape) + 16 * (-1)**0.5 * np.random.randn(*stft.shape)


pos_enc = [[(math.sin if i%2 == 0 else math.cos)(pos * math.pi / 2**(10 * (i//2)/256)) for i in range(512)]
           for pos in range(1024)]
pos_enc = 2*torch.tensor(pos_enc, dtype=dtype, device=device)


class AudioGen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trans = torch.nn.Transformer(512, num_encoder_layers=6, batch_first=True,
                                          dim_feedforward=2048, num_decoder_layers=6, activation=torch.nn.GELU(),
                                          norm_first=True, bias=True, dropout=0.1, dtype=dtype)
        # add convolution layer? how would you handle it being a mixture of real/imag parts?
        # probably group=2, but then i'd need to separate it into 2 channels which is a pain!

    def forward(self, audio, ctxlen):
        mask = torch.nn.Transformer.generate_square_subsequent_mask(ctxlen, device=device, dtype=dtype)
        trans = self.trans(audio, audio, src_mask=mask, tgt_mask=mask, memory_mask=mask,
                           src_is_causal=True, tgt_is_causal=True, memory_is_causal=True)
        return trans


model = AudioGen().to(device).train(True)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.SmoothL1Loss()

model.load_state_dict(torch.load("audiogen_big.txt"), strict=False)

fp = read("input.wav")  # MUST be SR = 16kHz
wav = np.array(fp[1], dtype=npdtype)
f, t, stft = signal.stft(wav, fs=SR, nperseg=FLEN, noverlap=OVERLAP, window=WIND)
stft = stft.transpose()  # by default, dimensions are (freq, time)
print(stft.shape, stft.dtype)
print(stft)
data = stft.view(np.float64).reshape((list(stft.shape)[0], 512))
data /= 1e-4 + np.abs(data).max(1, keepdims=True)
print(data.shape, data.dtype)
print(f"fourier shape {stft.shape}")


def write_out(net_repr, name="out.wav"):
    generation = net_repr.reshape(1024, 256, 2).astype(np.float32).view(np.complex64).reshape(1024, 256)
    print(generation)
    # print(data.shape)
    # generation = data[:1024]

    t2, istft = signal.istft(generation.transpose(), fs=SR, nperseg=FLEN, noverlap=OVERLAP, window=WIND)
    wavout = istft.reshape(-1)
    print(f"secs = {len(wavout) / SR}; timestep = {len(wavout) / (SR * 1024)}")
    write(name, SR, np.int16(32767 * wavout / np.max(np.abs(wavout))))


write_out(data[4096:4096+1024], "data.wav")

if __name__ == "__main__":
    for batch in range(chunk):
        opt.zero_grad()
        audio = []  # batch size
        expected_out = []
        for i in range(4):
            #
            start = 4096*i if random.randint(0, 100) <= 92 else random.randint(0, list(data.shape)[0] - 1026)
            audio.append(data[start:start+1024])
            expected_out.append(data[start:start+1024]) # todo: change back to offset
        audio_tensor = torch.tensor(np.array(audio), dtype=dtype, device=device)
        audio_out = model(audio_tensor + pos_enc, 1024)
        expected = torch.tensor(np.array(expected_out), dtype=dtype, device=device)
        # print(audio_out, '\n', expected_out)
        loss = loss_fn(audio_out, expected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

        loss = loss.item()
        if loss != loss:  # NaN
            break

        if (batch + 1) % 32 == 0:
            torch.save(model.state_dict(), "audiogen_big.txt")
            print(f"======= \\/ saved checkout \\/ ===========")

        print(f"{batch}/{chunk} {100*batch/chunk:.2f}% {audio_tensor.shape}: loss = {loss}")


