#%%
import cv2
from PIL import Image
from cv2 import sort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from trans_holo import asm

ex = 8
zi = 3E-2        # 伝搬距離
num_train = 600
num_test = 100
# batch_size = 64 #バッチサイズ
image_x, image_y = 28*ex, 28*ex
input_shape = (image_x, image_y, 1)
# epochs = 50
path = 'C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research'

print('start:trans_holo...')
h_train = np.zeros((num_train, image_x, image_y), dtype=np.complex64)
# h_test = np.zeros((num_test, image_x, image_y), dtype=np.complex64)

# 画像読み込み
for i in range(num_train):
    h_train[i] = np.loadtxt(path + f'\\mnist_data\\train\\mnist_train_{i}.csv', delimiter=',', dtype='complex64')

print(h_train[0])
print('Load Completed : h_train')

# for j in range(num_test):
#     h_test[j] = np.array(Image.open(path + f'\\mnist_data\\test\\mnist_test_{j}.png').convert('L'))
# print('Load Completed : h_test','\n')
# print(h_train.shape)

# 画像拡大
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train[:num_train]
# # x_test = x_test[:num_test]
# x_train = x_train.repeat(ex, axis=1).repeat(ex, axis=2)
# x_test = x_test.repeat(ex, axis=1).repeat(ex, axis=2)
# print(x_train.shape, '\n')

# ホログラム変換
# for i in range(num_train):
#     h_train[i] = asm(x_train[i], zi)
# for j in range(num_test):
#     h_test[j] = asm(x_test[j], z)
    
#%%
# DCGANネットワーク
img_rows = image_x
img_cols = image_y
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100


# 生成器（generator）
def build_generator(z_dim):

    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * 56 * 56, input_dim=z_dim))
    model.add(Reshape((56, 56, 256)))

    # Transposed convolution layer, from 56x56x256 into 112x112x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 112x112x128 to 112x112x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 112x112x64 to 224x224x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    # Output layer with tanh activation
    model.add(Activation('tanh'))

    return model

# 識別器
def build_discriminator(img_shape):

    model = Sequential()

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Fully connected layer
    model.add(Dense(128))
    
    model.add(Dropout(0.5))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))
    
    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    return model

# GANネットワーク
def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model

# 識別器の構築
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# 生成器の構築
generator = build_generator(z_dim)
# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

#%%
def sample_images(generator, iteration, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(path + f"\\picture\\DCGAN_predict{iteration+1}.png")


# GAN訓練
losses = []
accuracies = []
iteration_checkpoints = []


def train(f, iterations, batch_size, sample_interval):

    # Load the MNIST dataset
    # (X_train, _), (_, _) = mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    f = f.astype(np.float32) / 127.5 - 1.0
    # h_train = np.expand_dims(g_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images
        idx = np.random.randint(0, f.shape[0], batch_size)
        imgs = f[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  生成器の学習
        # ---------------------

       # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator, iteration)

#%% 
print("Training...")
iterations = 20000
batch_size = 64
sample_interval = 5000
train(h_train.real, iterations, batch_size, sample_interval) 