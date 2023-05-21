import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.datasets import mnist
from tensorflow_examples.models.pix2pix import pix2pix
import trans_holo
import culc_fid
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

# params
num = 1
ex = 8
size_y, size_x = 256, 256
LAMBDA = 10

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


# datasets - SVHN
dataset, metadata = tfds.load('svhn_cropped',
                              with_info=True, shuffle_files=True, as_supervised=True, batch_size=-1)

train_svhn = dataset['train'][0]
# test_svhn = dataset['test'][0]

train_svhn = np.stack(train_svhn)
# test_svhn = np.stack(test_svhn)
train_svhn = train_svhn[:num]
train_svhn = train_svhn.repeat(ex, axis=1).repeat(ex, axis=2)

for i in range(num):
    r = train_svhn[i, :, :, 0]
    g = train_svhn[i, :, :, 1]
    b = train_svhn[i, :, :, 2]
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        train_svhn[i, :, :, j] = v

# print(train_svhn.shape)
plt.imshow(train_svhn[0])
plt.show()

train_svhn = train_svhn.reshape(
    num, train_svhn.shape[1], train_svhn.shape[2], train_svhn.shape[3]
)

train_svhn = train_svhn.astype(np.float32)
train_svhn = tf.data.Dataset.from_tensor_slices(train_svhn)
# -------------------------------------------------------------------------


# datasets - MNIST
# print("Trans : mnist -> holo \n")
(m_train, _), (m_test, _) = mnist.load_data()
m_train = m_train[:num]
m_train = m_train.reshape(num, m_train.shape[1], m_train.shape[2], 1)
m_train = m_train.repeat(ex, axis=1).repeat(ex, axis=2).repeat(3, axis=3)
# m_test = m_test[:num_test]
# m_test = m_test.repeat(ex, axis=1).repeat(ex, axis=2)

# resize 256x256 (padding)

train_mnist = np.zeros((num, size_y, size_x, 3))
# test_mnist = np.zeros((num_test, size_y, size_x))
print(train_mnist.shape)

for i in range(num):
    train_mnist[i, 16:240, 16:240, :] = m_train[i]

# print(train_mnist.shape)
# plt.imshow(train_mnist[0])
# plt.show()

train_mnist = train_mnist.reshape(
    num, train_mnist.shape[1], train_mnist.shape[2], train_mnist.shape[3]
)

train_mnist = tf.data.Dataset.from_tensor_slices(train_mnist)

# ----------------------------------------------------------------------------


# 前処理関数
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


# 損失関数
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
    return LAMBDA * 0 * loss

# ------------------------------------------------------------------------------


# CycleGAN ---------------------------------------------------------------------
# Import Pix2Pix model
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Check points
checkpoint_path = "./checkpoints/svhn2mnist"

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


# Training ----------------------------------------------------------------------
EPOCHS = 10

train_svhn = train_svhn.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_mnist = train_mnist.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)


# 要素を順番に取り出す next(iter
sample_mnist = next(iter(train_svhn))
sample_holo = next(iter(train_mnist))

# to_svhn = generator_g(sample_mnist)
# to_mnist = generator_f(sample_holo)


def generate_images(model, test_input, num):
    prediction = model(test_input)

    # filename = f"C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\epoch_{num+1}.jpg"
    filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\epoch_{num+1}.jpg"

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
    for image_x, image_y in tf.data.Dataset.zip((train_svhn, train_svhn)):
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
    # Using a consistent image (sample_mnist) so that the progress of the model
    # is clearly visible.

    generate_images(generator_g, sample_mnist, epoch)

    # culclate FID
    real = train_svhn
    print("real -> ", np.stack(real).shape)

    fake = generator_g(sample_mnist)
    fake = np.stack(fake)
    fake = fake.reshape(fake.shape[0], 1, fake.shape[1],
                        fake.shape[2], fake.shape[3])
    print("fake -> ", fake.shape)

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


# 推移グラフ
plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(gen_g_loss)), gen_g_loss, marker='o', label='g_loss')
plt.plot(range(len(gen_f_loss)), gen_f_loss, marker='^', label='f_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

# plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\gen_loss.png")
plt.savefig(
    "C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\gen_loss.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(dis_x_loss)), dis_x_loss, marker='o', label='x_loss')
plt.plot(range(len(dis_y_loss)), dis_y_loss, marker='^', label='y_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

# plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\disc_loss.png")
plt.savefig(
    "C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\disc_loss.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('FID')
plt.plot(range(EPOCHS), FID, marker='o')
plt.legend(loc='best', fontsize=10)
plt.grid()

# plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\fid.png")
plt.savefig(
    "C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\svhn2mnist\\fid.png")
