import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import gc
# import bmp

ex = 8
z = 5E-3
num_train = 600
num_test = 100
lam = 532E-9
(x_train, _), (x_test, _) = mnist.load_data()


def calc_H(size_x, size_y, z, lam):
    dx = dy = 8E-6

    du = 1/(size_x*dx)
    dv = 1/(size_y*dy)

    H = np.zeros((size_x, size_y), dtype=np.complex64)

    for n in range(size_x):
        for m in range(size_y):
            w = np.sqrt((1 / lam)**2
                        - (du * (m - size_x / 2))**2
                        - (dv * (n - size_y / 2))**2)
            H[n][m] = np.cos(2 * np.pi * z * w) + \
                np.sin(2 * np.pi * z * w) * 1j

    H_sft = np.fft.fftshift(H)

    return H_sft


def padding(img):
    pad = np.zeros((img.shape[0]*2, img.shape[1]*2), dtype="float64")
    pad[int(pad.shape[0]/2-img.shape[0]/2):int(pad.shape[0]/2+img.shape[0]/2),
        int(pad.shape[1]/2-img.shape[1]/2):int(pad.shape[1]/2+img.shape[1]/2)] = img

    return pad


def trim(img):
    trim = np.zeros(
        (int(img.shape[0]/2), int(img.shape[1]/2)), dtype="float32")
    trim = img[int(img.shape[0]/4):int(img.shape[0]*3/4),
               int(img.shape[1]/4):int(img.shape[1]*3/4)]

    return trim


def asm(img, z, lam):
    # ゼロパディング
    pad = padding(img)

    # FFT変換
    G = np.zeros((pad.shape), dtype="complex64")
    G = np.fft.fft2(pad)
    max = G.real.max()
    min = G.real.min()
    # print("G:", max, min)

    # G×H 計算
    H = calc_H(G.shape[0], G.shape[1], z, lam)
    gh = G * H
    ift = np.fft.ifft2(gh)

    # 振幅型CGH
    amp = np.zeros((pad.shape), dtype="float32")
    amp = ift.real
    # print("ift:", max, min)

    # 正規化
    max = amp.max()
    min = amp.min()
    amp = (amp - min) / (max - min)

    # 位相型CGH
    phase = np.zeros((pad.shape), dtype="float32")
    phase = np.arctan2(ift.real, ift.imag)

    # 正規化
    max = phase.max()
    min = phase.min()
    phase = (phase - min) / (max - min)

    # 虚部
    imag = np.zeros((pad.shape), dtype="float32")
    imag = ift.imag * -1j
    # print("ift:", max, min)

    # 正規化
    max = imag.max()
    min = imag.min()
    imag = (imag - min) / (max - min)

    # トリミング
    amp = trim(amp)
    phase = trim(phase)
    imag = trim(imag) * -1j

    return amp, imag, phase


# img = x_train[0].repeat(8, 0).repeat(8, 1)
# print(img.shape)

# img = np.zeros((256, 256))
# img[126:130, 126:130] = 255

# plt.imshow(img)
# plt.gray()
# plt.show()

# holo = asm(img, z, lam)

# plt.imshow(holo.real)
# plt.gray()
# plt.show()
# rev = asm(holo, -z, lam)

# plt.imshow(rev.real)
# plt.gray()
# plt.show()

# print(rev[0, 0])


# plt.imshow(rev)
# plt.gray()
# plt.show()


# (width, height, colorTable, pixels) = bmp.read256BmpFile('rect.bmp')
# # print(width, height, colorTable, pixels)

# img = np.zeros((width, height))
# print(img.shape)
# n = 0
# for i in range(width):
#     for j in range(height):
#         img[i, j] = pixels[n] / 255  # [0, 1]スケーリング
#         n += 1

# # rect = asm(img, z)
# # H = calc_H(512, 512, 0.5)
# holo = asm(img, 0.5)
# # print("holo:", holo.max(), holo.min())

# colortable作成
# color = []
# for i in range(256):
#     for j in range(3):
#         color.append(i)
#     color.append(0)
# # print(color)


# # trainデータ
# for i in range(num_train):
#     img = x_train[i]

#     # 拡大
#     img = img.repeat(ex, axis=0)
#     img = img.repeat(ex, axis=1)
#     # print(img.shape)

#     width = img.shape[0]
#     height = img.shape[1]

#     # ASM
#     holo_train = asm(img, z)

#     pixels = []
#     for j in reversed(range(height)):
#         for k in range(width):
#             pixels.append(int(holo_train[j, k]))  # 1次元リスト変換
#     # print(len(pixels))

#     # 画像保存
#     # plt.imshow(holo_train)
#     # plt.gray()
#     # plt.show()

#     bmp.writeBmp(f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\mnist_data\\train\\holo_train_{i}.bmp",
#                  width, height, colorTables=color, pixels=pixels)

# # testデータ
# for i in range(num_test):
#     img = x_test[i]

#     # 拡大
#     img = img.repeat(ex, axis=0)
#     img = img.repeat(ex, axis=1)

#     width = img.shape[0]
#     height = img.shape[1]

#     # ASM
#     holo_test = asm(img, z)

#     pixels = []
#     for j in reversed(range(height)):
#         for k in range(width):
#             pixels.append(int(holo_test[j, k]))  # 1次元リスト変換
#     # print(len(pixels))

#     # 画像保存
#     # plt.imshow(holo_test)
#     # plt.gray()
#     # plt.show()

#     bmp.writeBmp(f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\mnist_data\\test\\holo_test_{i}.bmp",
#                  width, height, colorTables=color, pixels=pixels)
