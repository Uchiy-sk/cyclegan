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
z = 0.05
ex = 8
size_x, size_y = 256, 256
# lam = [640E-9, 532E-9, 447E-9]
lam = [532E-9, 532E-9, 532E-9]

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


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


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
checkpoint_path = "./checkpoints/gan_1"

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


# (x_train, _), (x_test, _) = mnist.load_data()
# m_test = x_test[:num].repeat(ex, axis=1).repeat(ex, axis=2)

# pad_test = np.zeros((num, size_y, size_x))
# for i in range(num):
#     pad_test[i, 16:240, 16:240] = m_test[i]

# plt.imshow(pad_test[0])
# plt.gray()
# plt.show()

# h_test = np.zeros_like(pad_test)
# for i in range(num):
#     h_test[i] = trans_holo.asm(pad_test[i], z, 532E-9)

# plt.imshow(h_test[0])
# plt.gray()
# plt.show()

# h_test = h_test.reshape(
#     num, h_test.shape[1], h_test.shape[2], 1).repeat(3, axis=3)
# pad_test = pad_test.reshape(
#     num, pad_test.shape[1], pad_test.shape[2], 1).repeat(3, axis=3)


# # print(h_test.shape)

# h_test = tf.data.Dataset.from_tensor_slices(h_test)
# h_test_data = h_test.cache().map(
#     preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=1)

# x_test = tf.data.Dataset.from_tensor_slices(pad_test)
# x_test_data = x_test.cache().map(
#     preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=1)

# for inp in x_test_data.take(1):
#     # print(inp)
#     generate_images(generator_g, inp, 0)

# for inp in h_test_data.take(1):
#     # print(inp)
#     generate_images(generator_f, inp, 1)

# # time->3.7s


# horse-----------------------------------------------------------------------------------------

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

test_horses, test_zebras = dataset['testA'], dataset['testB']

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

# create datasets
z = 0.02
lam = 532E-9

test_horses = np.stack(test_horses)
test_zebras = np.stack(test_zebras)

test_horse = test_horses[:1]
test_zebra = test_zebras[:1]

# gray -> R*0.3 + G*0.59 + B*0.11
for i in range(test_horse.shape[0]):
    r = (test_horse[i, :, :, :, 0] + 1)*127.5
    g = (test_horse[i, :, :, :, 1] + 1)*127.5
    b = (test_horse[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        test_horse[i, :, :, :, j] = v / 127.5 - 1

for i in range(test_zebra.shape[0]):
    r = (test_zebra[i, :, :, :, 0] + 1)*127.5
    g = (test_zebra[i, :, :, :, 1] + 1)*127.5
    b = (test_zebra[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        test_zebra[i, :, :, :, j] = v / 127.5 - 1

test_horse = test_horse.reshape(
    (test_horse.shape[0], test_horse.shape[2],
     test_horse.shape[3], test_horse.shape[4])
)
test_zebra = test_zebra.reshape(
    (test_zebra.shape[0], test_zebra.shape[2],
     test_zebra.shape[3], test_zebra.shape[4])
)

holo_horse = np.zeros_like(test_horse)
holo_zebra = np.zeros_like(test_zebra)


# image -> holo
for i in range(holo_horse.shape[0]):
    for j in range(holo_horse.shape[3]):
        holo_horse[i, :, :, j] = trans_holo.asm(
            test_horse[i, :, :, j], z, lam)[0] / 127.5 - 1
    print('.', end="")
print('')

for i in range(holo_zebra.shape[0]):
    for j in range(holo_zebra.shape[3]):
        holo_zebra[i, :, :, j] = trans_holo.asm(
            test_zebra[i, :, :, j], z, lam)[0] / 127.5 - 1
    print('.', end="")
print('')

holo_horse = holo_horse.reshape(
    holo_horse.shape[0], 1, holo_horse.shape[1], holo_horse.shape[2],  holo_horse.shape[3]
)
holo_zebra = holo_zebra.reshape(
    holo_zebra.shape[0], 1, holo_zebra.shape[1], holo_zebra.shape[2],  holo_zebra.shape[3]
)

test_horse = test_horse.reshape(
    test_horse.shape[0], 1, test_horse.shape[1], test_horse.shape[2],  test_horse.shape[3]
)
test_zebra = test_zebra.reshape(
    test_zebra.shape[0], 1, test_zebra.shape[1], test_zebra.shape[2],  test_zebra.shape[3]
)

# numpy -> datasets
test_horse = tf.data.Dataset.from_tensor_slices(test_horse)
test_zebra = tf.data.Dataset.from_tensor_slices(test_zebra)
holo_horse = tf.data.Dataset.from_tensor_slices(holo_horse)
holo_zebra = tf.data.Dataset.from_tensor_slices(holo_zebra)

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


for inp in test_horse.take(1):
    predict_1 = generator_g(inp)
    generate_images(generator_g, inp, 0)


# Check points ------------------------------------------------------------------
checkpoint_path = "./checkpoints/sampleamp"

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

print(generator_g.summary())
print(discriminator_x.summary())

for inp in test_horse.take(1):
    predict_2 = generator_g(inp)
    generate_images(generator_g, inp, 0)


ssim = tf.image.ssim(predict_1, predict_2, max_val=1.0)
psnr = tf.image.psnr(predict_1, predict_2, max_val=1.0)
print("ssim = ", ssim.numpy())
print("psnr = ", psnr.numpy())

# ssim =  [0.41723847]
# psnr =  [15.006164]

# ssim =  [0.7099735]
# psnr =  [17.483545]
