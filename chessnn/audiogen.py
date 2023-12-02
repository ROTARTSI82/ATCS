from audiogen_train import *

model = model.train(True)

start = 1600*3 # random.randint(0, list(data.shape)[0] - 1026)
# start = 4096# *random.randint(1, 3)


def gen():
    generation = data[start:start+64].tolist()
    ctx = 64
    while ctx < 1024:
        gen_tensor = torch.tensor(generation[-1024:], dtype=dtype, device=device)
        new = model(gen_tensor, ctx)
        ctx += 1
        print(len(generation))
        # print(new)
        generation.append(new[-1].tolist())
    return np.array(generation[-1024:], dtype=np.float32)


def preserve_test():
    buf = torch.tensor(data[start:start+1024], dtype=dtype, device=device)
    for i in range(1):
        buf = model(buf, 1024)
    return buf.numpy(force=True)


generation = gen()
print(generation)
write_out(generation)

write_out(preserve_test(), "preserve.wav")

