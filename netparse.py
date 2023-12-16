import struct
import cv2
import numpy as np

with open('trash.net', 'rb') as fp:
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

for i in range(layers):
    x, y = layerdims[i], layerdims[i+1]
    print(f"\n Layer {x}x{y}")

    images = [[] for _ in range(y)]
    for j in range(x):
        for k in range(y):
            w = struct.unpack(">f", data[:4])[0]
            if w < 0:
                images[k].append([-w, 0.0, 0.0])
            else:
                images[k].append([0.0, w, 0.0])
            data = data[4:]
            # print(f"\t{w}", end='')
        # print()
    if x == 128 * 128:
        for k in range(y):
            print('\n', np.array(images[k]).shape)
            arr = np.array(images[k]).reshape(128, 128, 3)
            mx = np.max(np.abs(arr.flatten()))
            arr *= 256 / mx
            print(arr.shape)
            cv2.imwrite(f"input{k}.png", arr)
