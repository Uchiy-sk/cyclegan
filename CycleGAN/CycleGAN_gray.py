# !pip install git+https://github.com/tensorflow/examples.git
# !pip install -U tfds-nightly

from random import sample
import tensorflow as tf
import tensorflow_datasets as tfds
# from keras.datasets import mnist
from tensorflow_examples.models.pix2pix import pix2pix
from trans_holo import asm
from culc_fid import get_fid
import numpy as np
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

print(type(os.getcwd()))

# Load Dataset
dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses = dataset['trainA']
test_horses = dataset['testA']

print(type(train_horses))

num_train = 1000
# num_test = 100

lam = 532E-9
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)  # Float型に変換
    image = (image / 127.5) - 1         # [0, 255] -> [-1, 1]
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    # print(image.shape)
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image

# cashe関数 -> キャッシュ化・処理高速化
# map関数 -> リストを全変換
# shuffle関数 -> 並び替え
# batch -> バッチサイズ毎に分割


train_horses = train_horses.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH_SIZE)\
    .shuffle(BUFFER_SIZE)

# test_horses = test_horses.cache().map(
#     preprocess_image_test, num_parallel_calls=AUTOTUNE)\
#     .batch(BATCH_SIZE)\
#     .shuffle(BUFFER_SIZE)

# print(type(train_horses))

z = 5E-3

train_horses = np.stack(train_horses)
# test_horses = np.stack(test_horses)
print(train_horses.shape)

# gray -> R*0.3 + G*0.59 + B*0.11
for i in range(num_train):
    r = (train_horses[i, :, :, :, 0] + 1)*127.5
    g = (train_horses[i, :, :, :, 1] + 1)*127.5
    b = (train_horses[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        train_horses[i, :, :, :, j] = v / 127.5 - 1

# for i in range(num_test):
#     r = (test_horses[i, :, :, :, 0] + 1)*127.5
#     g = (test_horses[i, :, :, :, 1] + 1)*127.5
#     b = (test_horses[i, :, :, :, 2] + 1)*127.5
#     v = 0.3*r + 0.59*g + 0.11*b
#     for j in range(3):
#         test_horses[i, :, :, :, j] = v / 127.5 - 1

train_horses = tf.data.Dataset.from_tensor_slices(train_horses)
# test_horses = tf.data.Dataset.from_tensor_slices(test_horses)

# Dataset -> numpy
train_holos = np.stack(train_horses)
# test_holos = np.stack(test_horses)


train_holos = train_holos[:num_train]
# test_holos = test_holos[:num_test]

# print(type(train_holos))
# <class 'numpy.ndarray'>
# print(train_holos.shape, test_holos.shape)
#(1067, 1, 256, 256, 3) (120, 1, 256, 256, 3)

# asm変換
train_holos = train_holos.reshape(
    (train_holos.shape[0], train_holos.shape[2],
     train_holos.shape[3], train_holos.shape[4])
)
# test_holos = test_holos.reshape(
#     (test_holos.shape[0], test_holos.shape[2],
#      test_holos.shape[3], test_holos.shape[4])
# )

# print(train_holos.shape, test_holos.shape)
# (1067, 256, 256, 3) (120, 256, 256, 3)

print("\ntrans_holo: train")
for i in tqdm(range(train_holos.shape[0])):
    train_holos[i, :, :, 0] = asm(                       # holo -> [0, 1]
        train_holos[i, :, :, 0], z, lam)[0] * 2 - 1      # holo -> [-1, 1]
    train_holos[i, :, :, 1] = asm(                       # holo -> [0, 1]
        train_holos[i, :, :, 0], z, lam)[1] * 2 - 1      # holo -> [-1, 1]
    train_holos[i, :, :, 2] = 0
    # print('.', end="")
print('')

# print("trans_holo: test")
# for i in tqdm(range(num_test)):
#     for j in range(test_holos.shape[3]):
#         test_holos[i, :, :, j] = trans_holo.asm(
#             test_holos[i, :, :, j], z, lam[j]) 
#     # print('.', end="")
# print('\n completed!!')

train_holos = train_holos.reshape(
    num_train, train_holos.shape[1], train_holos.shape[2],  train_holos.shape[3]
)
# test_holos = test_holos.reshape(
#     num_test, test_holos.shape[1], test_holos.shape[2],  test_holos.shape[3]
# )
print(train_holos[0, :, :, 0])
print(train_holos.shape)

# numpy -> Dataset
train_holos = tf.data.Dataset.from_tensor_slices(train_holos)
# test_holos = tf.data.Dataset.from_tensor_slices(test_holos)
print(type(train_holos))


train_holos = train_holos.cache().map(
    normalize, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH_SIZE)\
    .shuffle(BUFFER_SIZE)
# test_holos = test_holos.cache().map(
#     normalize, num_parallel_calls=AUTOTUNE)\
#     .batch(BATCH_SIZE)\
#     .shuffle(BUFFER_SIZE)


# 要素を順番に取り出す next(iter
sample_horse = next(iter(train_horses))
sample_holo = next(iter(train_holos))

# print('sample shape => ', sample_horse.shape)
# print('sample shape => ', sample_holo.shape)

# plt.subplot(121)
# plt.title('Horse')
# plt.imshow(sample_horse[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('Horse with random jitter')
# plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)

# plt.subplot(121)
# plt.title('holo')
# plt.imshow(sample_holo[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('holo with random jitter')
# plt.imshow(random_jitter(sample_holo[0]) * 0.5 + 0.5)
# plt.show()


# Import Pix2Pix model
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# to_holo = generator_g(sample_horse)
# to_horse = generator_f(sample_holo)

# plt.figure(figsize=(8, 8))
# contrast = 8

# imgs = [sample_horse, to_holo, sample_holo, to_horse]
# title = ['sample Horse', 'To holo', 'sample holo', 'To Horse']

# for i in range(len(imgs)):
#     plt.subplot(2, 2, i+1)
#     plt.title(title[i])
#     if i % 2 == 0:
#         plt.imshow(imgs[i][0] * 0.5 + 0.5)
#     else:
#         plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.savefig("sample")


# plt.figure(figsize=(8, 8))

# plt.subplot(121)
# plt.title('Is a real holo?')
# plt.imshow(discriminator_y(sample_holo)[0, ..., -1], cmap='RdBu_r')

# plt.subplot(122)
# plt.title('Is a real horse?')
# plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

# plt.show()


# Loss functions
LAMBDA = 5

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_mse(tf.ones_like(generated), generated)


# Cycle Loss
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

# Identity Loss


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Check points
checkpoint_path = "./checkpoints/gray_1000"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_path, max_to_keep=None)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


# Training
EPOCHS = 10


def generate_images(model, test_input, num=0):
    prediction = model(test_input)

    filename = f"C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\Epoch_{num}.jpg"
    # filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\Epoch_{num}.jpg"

    # print(type(filename))

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(filename)

FID = []
gen_g_loss = []
gen_f_loss = []
dis_x_loss = []
dis_y_loss = []


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        # print(type(gen_g_loss))

        total_cycle_loss = calc_cycle_loss(
            real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + \
            total_cycle_loss + identity_loss(real_y, same_y)

        total_gen_f_loss = gen_f_loss + \
            total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_holos)):
        total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(
            image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    gen_g_loss.append(total_gen_g_loss.numpy())
    gen_f_loss.append(total_gen_f_loss.numpy())
    dis_x_loss.append(disc_x_loss.numpy())
    dis_y_loss.append(disc_y_loss.numpy())

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images(generator_g, sample_horse, epoch+1)

    # culclate FID
    real = train_holos
    # print("real -> ", np.stack(real).shape)

    fake = generator_g(sample_horse)
    fake = np.stack(fake)
    fake = fake.reshape(fake.shape[0], 1, fake.shape[1],
                        fake.shape[2], fake.shape[3])
    # print("fake -> ", fake.shape)

    fake = tf.data.Dataset.from_tensor_slices(fake)

    fid = get_fid(real, fake)
    print("FID : ", fid)

    FID.append(fid)

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))


# Run the trained model on the test dataset

# for inp in test_horses.take(5):
#     generate_images(generator_g, inp)

# print(gen_g_loss)
# print(gen_f_loss)
# print(dis_x_loss)
# print(dis_y_loss)

# グラフ化
plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(gen_g_loss)), gen_g_loss, marker='o', label='g_loss')
plt.plot(range(len(gen_f_loss)), gen_f_loss, marker='^', label='f_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\generator_loss.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\generator_loss.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(dis_x_loss)), dis_x_loss, marker='o', label='x_loss')
plt.plot(range(len(dis_y_loss)), dis_y_loss, marker='^', label='y_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\discriminator_loss.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\discriminator_loss.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('FID')
plt.plot(range(EPOCHS), FID, marker='o')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\cyclegan_fid.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_gray\\cyclegan_fid.png")
