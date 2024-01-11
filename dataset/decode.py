import cv2
import numpy as np
import struct
import random


def dump(file, prefix):
    with open(file, 'rb') as fp:
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

        print(f"{np.mean(img)} {np.std(img)} [{np.min(img)}, {np.max(img)}]")
        img = img + 0.18 * np.random.randn(128, 128)
        for x in range(128):
            for y in range(128):
                if random.random() < 0.6:
                    img[x, y] = 0
        np.clip(img, 0.0, 1.0, img)

        # print(f"{img} -> {outp}")
        print(f"\t-> {np.mean(img)} {np.std(img)} [{np.min(img)}, {np.max(img)}]")

        cv2.imwrite(f"{prefix}{i//5}_{np.argmax(outp)}.bmp", 256 * img)


dump("train.data", "train/hand")
dump("validate.data", "validate/hand")
