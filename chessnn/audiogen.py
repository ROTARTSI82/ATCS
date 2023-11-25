from audiogen_train import *

model = model.eval()


def gen():
    start = random.randint(0, list(data.shape)[0] - 1026)
    generation = data[start:start+64].tolist()
    ctx = 64
    while ctx < 1024:
        gen_tensor = torch.tensor(generation, dtype=dtype, device=device)
        new = model(gen_tensor + pos_enc[:ctx], ctx)
        ctx += 1
        print(new.shape)
        # print(new)
        generation.append(new[-1].tolist())
    return np.array(generation, dtype=np.float32)


generation = gen()
print(generation)
write_out(generation)

