import cv2
import numpy as np
import struct

hw = 1024 + 256
hh = 1024 + 256

tw = 128
th = 128

blackout = 600


def encode(start, end, filename):
    db = b"ATNeuralDataset"
    db += struct.pack('>i', int(end - start + 1))
    db += struct.pack('>i', 128 * 128) # input dims
    db += struct.pack('>i', 5) # output dims
    cnt = 0

    for i in range(start, end+1):
        img = cv2.imread(f"raws/IMG_{i}.HEIC.jpeg", 0)
        print(np.max(img), np.min(img))
        print(img.shape, img)

        # div = 32
        # img = np.rint(256* np.exp(img / 256) / np.e).astype(int)

        # img = img // div * div + div // 2
        # img = img.astype(float) / 256.0

        # np.clip(img, 64, 256, out=img)

        print(np.max(img), np.min(img), img)

        # img[:blackout, :] = 0
        img[-blackout:, :] = 0
        # img[:, :blackout] = 0
        # img[:, -blackout:] = 0

        w, h = img.shape
        grid = np.mgrid[0:w:1j*w, 0:h:1j*h]

        col_x = np.sum(grid * img, axis=1)
        divisor = img.reshape(-1).sum()
        com = np.sum(col_x, axis=1) / divisor

        cx, cy = int(com[0]) - 300, int(com[1])
        img[img < 32] = 0
        img = cv2.GaussianBlur(img, (51, 51), 0)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        img = cv2.resize(img[cx-hw:cx+hw, cy-hh:cy+hh], (tw, th), interpolation=cv2.INTER_CUBIC)
        # pos = (pos[1], pos[0])
        print(img, cx, cy)
        # cv2.imshow('img', cv2.circle(img, (tw // 2, th // 2), radius=10, color=(0,0,0), thickness=10))

        clas = cnt % 5
        name = f"hand{cnt//5}_{clas + 1}"
        # cv2.imwrite(name + ".bmp", img)
        cnt += 1

        for b in img.reshape(-1):
            db += struct.pack(">f", float(b) / 256)
        for b in range(5):
            db += struct.pack('>f', float(1.0 if b == clas else 0))
        # cv2.waitKey()

    with open(filename, 'wb') as fp:
        fp.write(db)


encode(1899, 1899+24, "train.data")
encode(1899+25, 1928, "validate.data")
