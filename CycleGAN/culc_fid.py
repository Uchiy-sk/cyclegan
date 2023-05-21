import time
from random import sample
import os
import math
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras.datasets import mnist
import trans_holo
from tensorflow_examples.models.pix2pix import pix2pix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

# # パラメータ・変数設定 -------------------------------------------------------
num = 1
z = 0.005
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
# --------------------------------------------------------------------------

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
    # filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\amp_phase\\fid.jpg"

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
    # plt.show()
# -------------------------------------------------------------------------------


# Check points ------------------------------------------------------------------
checkpoint_path = "./checkpoints/sample"

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


# horse-----------------------------------------------------------------------------------------

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

test_horses, test_zebras = dataset['testA'], dataset['testB']

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)

# create datasets
z = 0.005
lam = 532E-9

test_horses = np.stack(test_horses)
test_zebras = np.stack(test_zebras)

test_horse = test_horses[:num]
test_zebra = test_zebras[:num]

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


# # image -> holo
# for i in tqdm(range(holo_horse.shape[0])):
#     for j in range(holo_horse.shape[3]):
#         holo_horse[i, :, :, j] = trans_holo.asm(
#             test_horse[i, :, :, j], z, lam)[0] / 127.5 - 1
#     # print('.', end="")
# print('')

# for i in tqdm(range(holo_zebra.shape[0])):
#     for j in range(holo_zebra.shape[3]):
#         holo_zebra[i, :, :, j] = trans_holo.asm(
#             test_zebra[i, :, :, j], z, lam)[0] / 127.5 - 1
#     # print('.', end="")
# print('')

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


# 画像推論

for inp in holo_horse.take(1):
    predict_zebra = generator_g(inp)
# generate_images(generator_g, inp, 0)

for inp in holo_zebra.take(1):
    predict_horse = generator_f(inp)
    # generate_images(generator_f, inp, 1)


# FID 計算 ------------------------------------------------------------------
real = holo_zebra
fake = np.stack(predict_zebra)
fake = fake.reshape(fake.shape[0], 1, fake.shape[1],
                    fake.shape[2], fake.shape[3])
fake = tf.data.Dataset.from_tensor_slices(fake)

# print(np.stack(real).shape)
# print(np.stack(fake).shape)


def compute_embeddings(data, count):
    inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights="imagenet",
                                                        pooling='avg')

    image_embeddings = []

    for _ in range(count):
        images = next(iter(data))
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)


def culc_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(
        axis=0), np.cov(real_embeddings, rowvar=False)

    mu2, sigma2 = generated_embeddings.mean(
        axis=0), np.cov(generated_embeddings, rowvar=False)

    # print("\nmu1 -> ", mu1)
    # print("sigma1 -> ", sigma1)
    # print("mu2 -> ", mu2)
    # print("sigma2 -> ", sigma2)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # print("ssdiff : ", ssdiff)

    # calculate sqrt of product between cov
    # print(sigma1.dot(sigma2))
    covmean = sqrtm(sigma1.dot(sigma2).real)
    # print("covmean_max : ", covmean.max())

    # check and correct imaginary numbers from sqrt
    # if np.iscomplexobj(covmean):
    #     covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_fid(real, fake):

    count = math.ceil(10/BATCH_SIZE)

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(real, count)

    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(fake, count)

    print(real_image_embeddings.shape, generated_image_embeddings.shape)

    FID = culc_fid(real_image_embeddings, generated_image_embeddings)
    return FID


# print("FID : ", get_fid(real, fake))

# amp_100 -> 220.883
# amp_1000 -> 214.990

# ---------------------------------------------------------------------------

# print(discriminator_x.summary())
