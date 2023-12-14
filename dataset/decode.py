import cv2
import numpy as np
import struct
import random

start = 1751
end = 1780

with open("validate.data", 'rb') as fp:
    data = fp.read()

assert data[:15] == b"ATNeuralDataset"

data = data[15:]
num = struct.unpack(">i", data[:4])[0]
in_dim = struct.unpack(">i", data[4:8])[0]
out_dim = struct.unpack(">i", data[8:12])[0]

print(f"{num} cases, {in_dim} in -> {out_dim} out")

data = data[12:]
for i in range(num):
    img = np.array([struct.unpack(">f", data[i*4:i*4+4])[0] for i in range(in_dim)]).reshape(128, 128)

    data = data[4*in_dim:]
    outp = np.array([struct.unpack(">f", data[i*4:i*4+4])[0] for i in range(out_dim)])
    data = data[4*out_dim:]

    print(f"{img} -> {outp}")

    img = img + 0.1 * np.random.randn(128, 128)

    for x in range(128):
        for y in range(128):
            if random.random() < 0.1:
                img[x, y] = 0

    cv2.imwrite(f"hand{i//5}_{np.argmax(outp)}.bmp", 256 * img)
