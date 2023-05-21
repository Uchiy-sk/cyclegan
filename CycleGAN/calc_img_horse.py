import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import trans_holo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.AUTOTUNE

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)


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


num = 1
z = 0.02
ex = 8
size_x, size_y = 256, 256
lam = [532E-9, 532E-9, 532E-9]

BUFFER_SIZE = 1000
BATCH_SIZE = 1


test_horses = dataset['testB']

test_horses = test_horses.cache().map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH_SIZE)


test_horses = np.stack(test_horses)[:num]

# gray -> R*0.3 + G*0.59 + B*0.11
for i in range(num):
    r = (test_horses[i, :, :, :, 0] + 1)*127.5
    g = (test_horses[i, :, :, :, 1] + 1)*127.5
    b = (test_horses[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        test_horses[i, :, :, :, j] = v / 127.5 - 1

test_horses = tf.data.Dataset.from_tensor_slices(test_horses)

test_horses = np.stack(test_horses)
test_horses = test_horses[:num]
test_horses = test_horses.reshape(
    test_horses.shape[1], test_horses.shape[2], test_horses.shape[3], test_horses.shape[4]
)
print(test_horses[0].max(), test_horses[0].min())

plt.figure(figsize=(12, 8))
plt.title('horse')
plt.imshow(test_horses[0] * 0.5 + 0.5)
plt.axis('off')
plt.show()

print('\nasm : horse -> holo\n')

test_holos = np.zeros_like(test_horses)

print(test_holos.shape)

for i in range(num):
    for j in range(test_holos.shape[3]):
        test_holos[i, :, :, j] = trans_holo.asm(
            test_horses[i, :, :, j], z, lam[j]) / 127.5 - 1

test_holos = test_holos.reshape(
    num, test_holos.shape[1], test_holos.shape[2],  test_holos.shape[3]
)

print(test_holos[0].max(), test_holos[0].min())

plt.figure(figsize=(12, 8))
plt.title('holo')
plt.imshow(test_holos[0] * 0.5 + 0.5)
plt.axis('off')
plt.show()

print('\nasm : holo -> rev\n')

test_revs = np.zeros_like(test_holos)

for i in range(num):
    for j in range(test_revs.shape[3]):
        test_revs[i, :, :, j] = trans_holo.asm(
            test_holos[i, :, :, j], -z, lam[j]) / 127.5 - 1

plt.figure(figsize=(12, 8))
plt.title('rev')
plt.imshow(test_revs[0] * 0.5 + 0.5)
plt.axis('off')
plt.show()


print(tf.image.ssim())


# print('\nasm : rev -> rev_holo\n')
# test_cycs = np.zeros_like(test_revs)

# for i in range(num):
#     for j in range(test_cycs.shape[3]):
#         test_cycs[i, :, :, j] = trans_holo.asm(
#             test_revs[i, :, :, j], z, lam[j]) / 127.5 - 1

# plt.figure(figsize=(12, 8))
# plt.title('cyc')
# plt.imshow(test_cycs[0] * 0.5 + 0.5)
# plt.axis('off')
# plt.show()
