import struct
import cv2
import numpy as np

with open('network2.net', 'rb') as fp:
    data = fp.read()

assert data[:15] == b"ATNeuralNetwork"
data = data[15:]

layers = struct.unpack(">i", data[:4])[0]
in_dim = struct.unpack(">i", data[4:8])[0]
data = data[8:]

layerdims = [in_dim, ]
for i in range(layers):
    layerdims.append(struct.unpack(">i", data[:4])[0])
    data = data[4:]

print(f"{layers} layers, {layerdims}")


def statsof(arr):
    return f"{np.mean(arr):.6f} {np.std(arr):.6f} [{np.min(arr):.6f} {np.max(arr):.6f}]"


for i in range(layers):
    x, y = layerdims[i], layerdims[i+1]
    print(f"\n Layer {i} ({x}x{y})")

    # opencv is BGR
    negs = np.array([0, 0, 1], dtype=">f")
    pos = np.array([0, 1, 0], dtype=">f")

    raw = np.frombuffer(data[:x*y*4], dtype=">f").reshape((x, y))

    print(f"\teinsumming! stats: {statsof(raw)}")
    # print(raw)
    mask = (raw < 0).astype(float)
    images = np.einsum("xy,xy,c->yxc", -mask, raw, negs) + np.einsum("xy,xy,c->yxc", 1-mask, raw, pos)
    data = data[x*y*4:]

    write_dreams = True
    write_matrices = True
    if write_dreams and x == 128 * 128:
        print("\twriting input pngs")
        for k in range(y):
            # print(f"\tinput {k} {statsof(images[k])}")
            arr = images[k].reshape((128, 128, 3))
            mx = np.max(np.abs(arr.flatten()))
            arr *= 256 / mx
            cv2.imwrite(f"vis/input{k}.png", arr)

    if write_matrices:
        print(images.shape)
        mx = np.max(np.abs(images.flatten()))
        images *= 256 / mx
        cv2.imwrite(f"vis/layer{i}_{x}x{y}.png", images)
