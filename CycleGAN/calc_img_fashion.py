import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import trans_holo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.AUTOTUNE


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


z = 0.05
ex = 8
size_x, size_y = 256, 256
lam = [640E-9, 532E-9, 447E-9]

BUFFER_SIZE = 1000
BATCH_SIZE = 1


(x_train, _), (x_test, _) = fashion_mnist.load_data()
m_test = x_test[0].repeat(ex, axis=0).repeat(ex, axis=1)

pad_test = np.zeros((size_y, size_x))
pad_test[16:240, 16:240] = m_test

plt.imshow(pad_test)
plt.axis('off')
plt.gray()
plt.show()

h_test = np.zeros_like(pad_test)

start = time.perf_counter()
h_test = trans_holo.asm(pad_test, z, 532E-9)

end = time.perf_counter()
print('calc time -> ', end - start)
# 1.6s

plt.imshow(h_test)
plt.axis('off')
plt.gray()
plt.show()


rev_test = np.zeros_like(h_test)
rev_test = trans_holo.asm(h_test, -z, 532E-9)

# plt.imshow(rev_test)
# plt.axis('off')
# plt.gray()
# plt.show()