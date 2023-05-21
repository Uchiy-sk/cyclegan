import time
from random import sample
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras.datasets import mnist
import trans_holo
from tensorflow_examples.models.pix2pix import pix2pix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

# !pip install git+https://github.com/tensorflow/examples.git


num = 1
z = 0.005
ex = 8
size_x, size_y = 256, 256
# lam = [640E-9, 532E-9, 447E-9]
lam = 532E-9

BUFFER_SIZE = 1000
BATCH_SIZE = 1


OUTPUT_CHANNELS = 3
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# 関数定義 ------------------------------------------------------------------------
def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[size_y, size_x, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    # print(image.shape)
    image = tf.image.resize(image, [256, 256],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    # image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = normalize(image)
    return image


def generate_images(model, test_input, num):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.savefig(filename)
    plt.show()


def generate_images(model, test_input, num=0):

    start = time.perf_counter()

    prediction = model(test_input)

    end = time.perf_counter()
    print('predict time -> ', end - start)
    # 3.465 s

    # filename = f"C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_fashion\\predict_f_{num}.jpg"
    # filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_fashion\\predict_g_{num}.jpg"

    plt.figure(figsize=(12, 8))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow((display_list[i] + 1) * 0.5)
        plt.axis('off')
    # plt.savefig(filename)
    plt.show()


# Check points ------------------------------------------------------------------
checkpoint_path = "./checkpoints/num_0.005"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# max_to_keep -> 最新のパラメータ5つを保存

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# --------------------------------------------------------------------------------


# mnist ----------------------------------------------------------------------------------------------


# データセット（224x224)
print("Trans : mnist -> holo \n")
(m_train, _), (_, _) = mnist.load_data()

# mnist_train, holo_train = dataset['trainA'], dataset['trainB']
# mnist_test, holo_test = dataset['testA'], dataset['testB']

m_train = m_train[:num]
m_train = m_train.repeat(ex, axis=1).repeat(ex, axis=2)
# m_test = m_test[:num_test]
# m_test = m_test.repeat(ex, axis=1).repeat(ex, axis=2)
#

# resize 256x256 (padding)

mnist_train = np.zeros((num, size_y, size_x))
# mnist_test = np.zeros((num_test, size_y, size_x))
print(mnist_train.shape)

for i in range(num):
    mnist_train[i, 16:240, 16:240] = m_train[i]

# for j in range(num_test):
#     mnist_test[j, 16:240, 16:240] = m_test[j]


# holo画像 作成
holo_train = np.zeros_like(mnist_train)
# holo_test = np.zeros_like(mnist_test)

for i in range(num):
    holo_train[i] = trans_holo.asm(mnist_train[i], z, lam)[0] * 2 - 1
    print('.', end="")
print('')


# for j in range(num_test):
#     holo_test[j] = trans_holo.asm(mnist_test[j], z, lam)
#     print('.', end="")
# print('\n')

# resize -> 3x256x256x3
mnist_train = mnist_train.reshape(
    num, mnist_train.shape[1], mnist_train.shape[2], 1)
holo_train = holo_train.reshape(
    num, holo_train.shape[1], holo_train.shape[2], 1)
# mnist_test = mnist_test.reshape(
#     num_test, mnist_test.shape[1], mnist_test.shape[2], 1)
# holo_test = holo_test.reshape(
#     num_test, holo_test.shape[1], holo_test.shape[2], 1)

mnist_train_ex = mnist_train.repeat(3, axis=3)
holo_train_ex = holo_train.repeat(3, axis=3)
# mnist_test_ex = mnist_test.repeat(3, axis=3)
# holo_test_ex = holo_test.repeat(3, axis=3)

# print("mnist_test_ex -> ", mnist_test_ex.shape)
# [num, 256, 256, 3]

plt.imshow(holo_train[0])
plt.show()
# print(h_test.shape)

h_test = tf.data.Dataset.from_tensor_slices(holo_train_ex)
h_test_data = h_test.cache().map(
    preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=1)

x_test = tf.data.Dataset.from_tensor_slices(mnist_train_ex)
x_test_data = x_test.cache().map(
    preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=1)

for inp in x_test_data.take(1):
    # print(inp)
    generate_images(generator_g, inp, 0)

for inp in h_test_data.take(1):
    # print(inp)
    generate_images(generator_f, inp, 1)

# time->3.7s


# # horse-----------------------------------------------------------------------------------------

# dataset, metadata = tfds.load('cycle_gan/horse2zebra',
#                               with_info=True, as_supervised=True)

# test_horses, test_zebras = dataset['testA'], dataset['testB']

# test_horses = test_horses.map(
#     preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

# test_zebras = test_zebras.map(
#     preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

# # create datasets
# z = 0.005
# lam = 532E-9

# test_horses = np.stack(test_horses)
# test_zebras = np.stack(test_zebras)

# test_horse = test_horses[:1]
# test_zebra = test_zebras[:1]

# # gray -> R*0.3 + G*0.59 + B*0.11
# for i in range(test_horse.shape[0]):
#     r = (test_horse[i, :, :, :, 0] + 1)*127.5
#     g = (test_horse[i, :, :, :, 1] + 1)*127.5
#     b = (test_horse[i, :, :, :, 2] + 1)*127.5
#     v = 0.3*r + 0.59*g + 0.11*b
#     for j in range(3):
#         test_horse[i, :, :, :, j] = v / 127.5 - 1

# for i in range(test_zebra.shape[0]):
#     r = (test_zebra[i, :, :, :, 0] + 1)*127.5
#     g = (test_zebra[i, :, :, :, 1] + 1)*127.5
#     b = (test_zebra[i, :, :, :, 2] + 1)*127.5
#     v = 0.3*r + 0.59*g + 0.11*b
#     for j in range(3):
#         test_zebra[i, :, :, :, j] = v / 127.5 - 1

# test_horse = test_horse.reshape(
#     (test_horse.shape[0], test_horse.shape[2],
#      test_horse.shape[3], test_horse.shape[4])
# )
# test_zebra = test_zebra.reshape(
#     (test_zebra.shape[0], test_zebra.shape[2],
#      test_zebra.shape[3], test_zebra.shape[4])
# )

# holo_horse = np.zeros_like(test_horse)
# holo_zebra = np.zeros_like(test_zebra)


# # image -> holo
# for i in range(holo_horse.shape[0]):
#     for j in range(holo_horse.shape[3]):
#         holo_horse[i, :, :, j] = trans_holo.asm(
#             test_horse[i, :, :, j], z, lam)[0] * 2 - 1
#     print('.', end="")
# print('')

# for i in range(holo_zebra.shape[0]):
#     for j in range(holo_zebra.shape[3]):
#         holo_zebra[i, :, :, j] = trans_holo.asm(
#             test_zebra[i, :, :, j], z, lam)[0] * 2 - 1
#     print('.', end="")
# print('')

# holo_horse = holo_horse.reshape(
#     holo_horse.shape[0], 1, holo_horse.shape[1], holo_horse.shape[2],  holo_horse.shape[3]
# )
# holo_zebra = holo_zebra.reshape(
#     holo_zebra.shape[0], 1, holo_zebra.shape[1], holo_zebra.shape[2],  holo_zebra.shape[3]
# )

# test_horse = test_horse.reshape(
#     test_horse.shape[0], 1, test_horse.shape[1], test_horse.shape[2],  test_horse.shape[3]
# )
# test_zebra = test_zebra.reshape(
#     test_zebra.shape[0], 1, test_zebra.shape[1], test_zebra.shape[2],  test_zebra.shape[3]
# )

# # numpy -> datasets
# test_horse = tf.data.Dataset.from_tensor_slices(test_horse)
# test_zebra = tf.data.Dataset.from_tensor_slices(test_zebra)
# holo_horse = tf.data.Dataset.from_tensor_slices(holo_horse)
# holo_zebra = tf.data.Dataset.from_tensor_slices(holo_zebra)

# for inp in test_horse.take(1):
#     predict = generator_g(inp)
#     generate_images(generator_g, inp, 0)

# # for inp in holo_zebra.take(1):
# #     predict = generator_f(inp)
# #     generate_images(generator_f, inp, 1)

# rev = np.zeros_like(predict)

# for i in range(3):
#     rev[0, :, :, i] = trans_holo.asm(predict[0, :, :, i], -z, lam) / 255

# # print(predict.max(), predict.min())

# plt.figure(figsize=(12, 8))
# plt.imshow((rev[0]))
# plt.axis('off')
# plt.show()

# real = np.stack(holo_horse)
# fake = predict

# ssim = tf.image.ssim(real, fake, max_val=1.0)
# psnr = tf.image.psnr(real, fake, max_val=1.0)
# print("ssim = ", ssim.numpy()[0, 0])
# print("psnr = ", psnr.numpy()[0, 0])


# for inp in test_horse.take(1):
#     predict_1 = generator_g(inp)
#     generate_images(generator_g, inp, 0)


# print(generator_g.summary())
# print(discriminator_x.summary())

# for inp in test_horse.take(1):
#     predict_2 = generator_g(inp)
#     generate_images(generator_g, inp, 0)


# ssim = tf.image.ssim(predict_1, predict_2, max_val=1.0)
# psnr = tf.image.psnr(predict_1, predict_2, max_val=1.0)
# print("ssim = ", ssim.numpy())
# print("psnr = ", psnr.numpy())

# # ssim =  [0.41723847]
# # psnr =  [15.006164]

# # ssim =  [0.7099735]
# # psnr =  [17.483545]
