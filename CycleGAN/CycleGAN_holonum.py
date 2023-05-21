# !pip install git+https://github.com/tensorflow/examples.git
# !pip install -U tfds-nightly

# from random import sample
import tensorflow as tf
import keras
from keras.datasets import mnist
from tensorflow_examples.models.pix2pix import pix2pix
import trans_holo
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

# Load Dataset
# dataset, metadata = tfds.load('cycle_gan/mnist2holo',
#                               with_info=True, as_supervised=True)

# mnist_train, holo_train = dataset['trainA'], dataset['trainB']
# mnist_test, holo_test = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
lam = 532E-9

num_train = 10
# num_test = 10

ex = 8
z = 5E-3
size_x, size_y = 256, 256


# データセット（224x224)
print("Trans : mnist -> holo \n")
(m_train, _), (_, _) = mnist.load_data()

# mnist_train, holo_train = dataset['trainA'], dataset['trainB']
# mnist_test, holo_test = dataset['testA'], dataset['testB']

m_train = m_train[:num_train]
m_train = m_train.repeat(ex, axis=1).repeat(ex, axis=2)
# m_test = m_test[:num_test]
# m_test = m_test.repeat(ex, axis=1).repeat(ex, axis=2)
#

# resize 256x256 (padding)

mnist_train = np.zeros((num_train, size_y, size_x))
# mnist_test = np.zeros((num_test, size_y, size_x))
print(mnist_train.shape)

for i in range(num_train):
    mnist_train[i, 16:240, 16:240] = m_train[i]

# for j in range(num_test):
#     mnist_test[j, 16:240, 16:240] = m_test[j]


# holo画像 作成
holo_train = np.zeros_like(mnist_train)
# holo_test = np.zeros_like(mnist_test)

for i in range(num_train):
    holo_train[i] = trans_holo.asm(mnist_train[i], z, lam)
    print('.', end="")
print('')

# for j in range(num_test):
#     holo_test[j] = trans_holo.asm(mnist_test[j], z, lam)
#     print('.', end="")
# print('\n')

# resize -> 3x256x256x3
mnist_train = mnist_train.reshape(
    num_train, mnist_train.shape[1], mnist_train.shape[2], 1)
holo_train = holo_train.reshape(
    num_train, holo_train.shape[1], holo_train.shape[2], 1)
# mnist_test = mnist_test.reshape(
#     num_test, mnist_test.shape[1], mnist_test.shape[2], 1)
# holo_test = holo_test.reshape(
#     num_test, holo_test.shape[1], holo_test.shape[2], 1)

mnist_train_ex = mnist_train.repeat(3, axis=3)
holo_train_ex = holo_train.repeat(3, axis=3)
# mnist_test_ex = mnist_test.repeat(3, axis=3)
# holo_test_ex = holo_test.repeat(3, axis=3)

plt.imshow(holo_train_ex[0])
plt.gray()
plt.show()

# print("mnist_test_ex -> ", mnist_test_ex.shape)
# [num, 256, 256, 3]

# -------------------------


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

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


# np.array -> tf.dataオブジェクト変換
mnist_train_ex = tf.data.Dataset.from_tensor_slices(mnist_train_ex)
# mnist_test_ex = tf.data.Dataset.from_tensor_slices(mnist_test_ex)
holo_train_ex = tf.data.Dataset.from_tensor_slices(holo_train_ex)
# holo_test_ex = tf.data.Dataset.from_tensor_slices(holo_test_ex)

mnist_train_data = mnist_train_ex.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH_SIZE)\
    .shuffle(BUFFER_SIZE)

# mnist_test_data = mnist_test_ex.cache().map(
#     preprocess_image_train, num_parallel_calls=AUTOTUNE)\
#     .batch(BATCH_SIZE)\
#     .shuffle(BUFFER_SIZE)

holo_train_data = holo_train_ex.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH_SIZE)\
    .shuffle(BUFFER_SIZE)

# holo_test_data = holo_test_ex.cache().map(
#     preprocess_image_train, num_parallel_calls=AUTOTUNE)\
#     .batch(BATCH_SIZE)\
#     .shuffle(BUFFER_SIZE)

print(type(mnist_train_data))

# 要素を順番に取り出す next(iter
sample_mnist = next(iter(mnist_train_data))
sample_holo = next(iter(holo_train_data))

print('sample shape => ', sample_holo.shape)

# plt.subplot(121)
# plt.title('mnist')
# plt.imshow(sample_mnist[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('mnist with random jitter')
# plt.imshow(random_jitter(sample_mnist[0]) * 0.5 + 0.5)

# plt.subplot(121)
# plt.title('holo')
# plt.imshow(sample_holo[0] * -0.5 + 0.5)

# plt.subplot(122)
# plt.title('holo with random jitter')
# plt.imshow(random_jitter(sample_holo[0]) * -0.5 + 0.5)


# Import Pix2Pix model
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# to_holo = generator_g(sample_mnist)
# to_mnist = generator_f(sample_holo)

# plt.figure(figsize=(8, 8))
# contrast = 8

# imgs = [sample_mnist, to_holo, sample_holo, to_mnist]
# title = ['mnist', 'To holo', 'holo', 'To mnist']

# for i in range(len(imgs)):
#     plt.subplot(2, 2, i+1)
#     plt.title(title[i])
#     if i % 2 == 0:
#         plt.imshow(imgs[i][0] * 0.5 + 0.5)
#     else:
#         plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()


# plt.figure(figsize=(8, 8))

# plt.subplot(121)
# plt.title('Is a real holo?')
# plt.imshow(discriminator_y(sample_holo)[0, ..., -1], cmap='RdBu_r')

# plt.subplot(122)
# plt.title('Is a real mnist?')
# plt.imshow(discriminator_x(sample_mnist)[0, ..., -1], cmap='RdBu_r')

# plt.show()


# Loss functions
LAMBDA = 10

loss_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_mse = tf.keras.losses.MeanSquaredError()
# loss_ssim = tf.image.ssim()


def loss_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)
                                          # filter_size=11
                                          # filter_sigma=1.5
                                          # k1=0.01
                                          # k2=0.03


def discriminator_loss(real, generated):
    real_loss = loss_bce(tf.ones_like(real), real)

    generated_loss = loss_ssim(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    # print("disc loss -> ", total_disc_loss.eval(session=tf.compat.v1.Session()))

    return total_disc_loss * 0.5


def generator_loss(generated):
    gen_loss = loss_ssim(tf.ones_like(generated), generated)
    # print("gen loss -> ", gen_loss.eval(session=tf.compat.v1.Session()))
    return gen_loss


# Cycle Loss
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))  # 平均値の算出

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
checkpoint_path = "./checkpoints/train"

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
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')


# Training
EPOCHS = 10


def generate_images(model, test_input, num):
    prediction = model(test_input)

    filename = f"C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\epoch_{num+1}.jpg"
    # filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\epoch_{num+1}.jpg"

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
    for image_x, image_y in tf.data.Dataset.zip((mnist_train_data, holo_train_data)):
        total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    gen_g_loss.append(total_gen_g_loss.numpy())
    gen_f_loss.append(total_gen_f_loss.numpy())
    dis_x_loss.append(disc_x_loss.numpy())
    dis_y_loss.append(disc_y_loss.numpy())

    clear_output(wait=True)
    # Using a consistent image (sample_mnist) so that the progress of the model
    # is clearly visible.

    generate_images(generator_f, sample_holo, epoch)

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))

# Run the trained model on the test dataset
# for inp in mnist_test_data.take(5):
#     generate_images(generator_g, inp)


plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(gen_g_loss)), gen_g_loss, marker='o', label='g_loss')
plt.plot(range(len(gen_f_loss)), gen_f_loss, marker='^', label='f_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\gen.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\gen.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(dis_x_loss)), dis_x_loss, marker='o', label='x_loss')
plt.plot(range(len(dis_y_loss)), dis_y_loss, marker='^', label='y_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\disc.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\z=0.005(1000)\\disc.png")
