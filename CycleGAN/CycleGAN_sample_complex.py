# !pip install git+https://github.com/tensorflow/examples.git
# !pip install -U tfds-nightly

from random import sample
import tensorflow as tf
import tensorflow_datasets as tfds
# from keras.datasets import mnist
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import trans_holo
from culc_fid import get_fid
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.AUTOTUNE

# Load Dataset
dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

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
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_zebras = train_zebras.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

print(type(train_horses))


# create datasets
z = 0.005
lam = 532E-9

holo_horses = np.stack(train_horses)
holo_zebras = np.stack(train_zebras)

holo_horses = holo_horses[:1000]
holo_zebras = holo_zebras[:1000]


# gray -> R*0.3 + G*0.59 + B*0.11
for i in range(holo_horses.shape[0]):
    r = (holo_horses[i, :, :, :, 0] + 1)*127.5
    g = (holo_horses[i, :, :, :, 1] + 1)*127.5
    b = (holo_horses[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        holo_horses[i, :, :, :, j] = v

for i in range(holo_zebras.shape[0]):
    r = (holo_zebras[i, :, :, :, 0] + 1)*127.5
    g = (holo_zebras[i, :, :, :, 1] + 1)*127.5
    b = (holo_zebras[i, :, :, :, 2] + 1)*127.5
    v = 0.3*r + 0.59*g + 0.11*b
    for j in range(3):
        holo_zebras[i, :, :, :, j] = v              # holo -> [0, 255]


holo_horses = holo_horses.reshape(
    (holo_horses.shape[0], holo_horses.shape[2],
     holo_horses.shape[3], holo_horses.shape[4])
)
holo_zebras = holo_zebras.reshape(
    (holo_zebras.shape[0], holo_zebras.shape[2],
     holo_zebras.shape[3], holo_zebras.shape[4])
)

# image -> holo
for i in range(holo_horses.shape[0]):
    holo_horses[i, :, :, 0] = trans_holo.asm(           # holo -> [0, 1]
        holo_horses[i, :, :, 0], z, lam)[0] * 2 - 1      # holo -> [-1, 1]
    holo_horses[i, :, :, 1] = trans_holo.asm(           # holo -> [0, 1]
        holo_horses[i, :, :, 0], z, lam)[1] * 2 - 1      # holo -> [-1, 1]
    holo_horses[i, :, :, 2] = 0
    print('.', end="")
print('')

for i in range(holo_zebras.shape[0]):
    holo_zebras[i, :, :, 0] = trans_holo.asm(
        holo_zebras[i, :, :, 0], z, lam)[0] * 2 - 1
    holo_zebras[i, :, :, 1] = trans_holo.asm(
        holo_zebras[i, :, :, 0], z, lam)[1] * 2 - 1
    holo_zebras[i, :, :, 2] = 0
    print('.', end="")
print('')

holo_horses = holo_horses.reshape(
    holo_horses.shape[0], 1, holo_horses.shape[1], holo_horses.shape[2],  holo_horses.shape[3]
)
holo_zebras = holo_zebras.reshape(
    holo_zebras.shape[0], 1, holo_zebras.shape[1], holo_zebras.shape[2],  holo_zebras.shape[3]
)

# numpy -> datasets
holo_horses = tf.data.Dataset.from_tensor_slices(holo_horses)
holo_zebras = tf.data.Dataset.from_tensor_slices(holo_zebras)


# 要素を順番に取り出す next(iter
sample_horse = next(iter(holo_horses))
sample_zebra = next(iter(holo_zebras))

print('sample shape => ', sample_horse.shape)
print('sample shape => ', sample_zebra.shape)

# plt.subplot(121)
# plt.title('Horse')
# plt.imshow(sample_horse[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('Horse with random jitter')
# plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)

# plt.subplot(121)
# plt.title('Zebra')
# plt.imshow(sample_zebra[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('Zebra with random jitter')
# plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)


# Import Pix2Pix model
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# to_zebra = generator_g(sample_horse)
# to_horse = generator_f(sample_zebra)

# plt.figure(figsize=(8, 8))
# contrast = 8

# imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
# title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

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
# plt.title('Is a real zebra?')
# plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

# plt.subplot(122)
# plt.title('Is a real horse?')
# plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

# plt.show()


# Loss functions
LAMBDA = 10

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
checkpoint_path = "./checkpoints/complex"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


# Training
EPOCHS = 10


def generate_images(model, test_input, num):
    filename = f"C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_epoch_{num}.jpg"
    # filename = f"C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_epoch_{num}.jpg"

    prediction = model(test_input)

    print((prediction[0, :, :, 0] + 1) * 127.5)

    plt.figure(figsize=(12, 8))

    display_list = [test_input[0, :, :, 0],
                    (prediction[0, :, :, 0] + 1) * 127.5]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.gray()
        plt.axis('off')
    plt.savefig(filename)

FID_horse = []
FID_zebra = []
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
    for image_x, image_y in tf.data.Dataset.zip((holo_horses, holo_zebras)):
        total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(
            image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))

#
    generate_images(generator_g, sample_horse, epoch+1)
#
    gen_g_loss.append(total_gen_g_loss.numpy())
    gen_f_loss.append(total_gen_f_loss.numpy())
    dis_x_loss.append(disc_x_loss.numpy())
    dis_y_loss.append(disc_y_loss.numpy())
#

    # culclate FID
    real_g = holo_zebras
    real_f = holo_horses
    print("real_g -> ", np.stack(real_g).shape)
    print("real_f -> ", np.stack(real_f).shape)

    fake_g = generator_g(sample_horse)
    fake_f = generator_f(sample_zebra)
    fake_g = np.stack(fake_g)
    fake_f = np.stack(fake_f)
    fake_g = fake_g.reshape(fake_g.shape[0], 1, fake_g.shape[1],
                        fake_g.shape[2], fake_g.shape[3])
    fake_f = fake_f.reshape(fake_f.shape[0], 1, fake_f.shape[1],
                        fake_f.shape[2], fake_f.shape[3])
    print("fake_g -> ", fake_g.shape)
    print("fake_f -> ", fake_f.shape)
    
    fake_g = tf.data.Dataset.from_tensor_slices(fake_g)
    fake_f = tf.data.Dataset.from_tensor_slices(fake_f)

    fid_horse = get_fid(real_g, fake_g)
    fid_zebra = get_fid(real_f, fake_f)
    print("FID_g : ", fid_horse)
    print("FID_f : ", fid_zebra)
    FID_horse.append(fid_horse)
    FID_zebra.append(fid_zebra)





# Run the trained model on the test dataset
# for inp in test_horses.take(5):
#     generate_images(generator_g, inp)

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(gen_g_loss)), gen_g_loss, marker='o', label='g_loss')
plt.plot(range(len(gen_f_loss)), gen_f_loss, marker='^', label='f_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_gen.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_gen.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(dis_x_loss)), dis_x_loss, marker='o', label='x_loss')
plt.plot(range(len(dis_y_loss)), dis_y_loss, marker='^', label='y_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_disc.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_disc.png")

plt.figure(figsize=(8, 6))

plt.xlabel('epoch')
plt.ylabel('FID')
plt.plot(range(len(FID_horse)), FID_horse, marker='o', label='g_loss')
plt.plot(range(len(FID_zebra)), FID_zebra, marker='^', label='f_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.savefig("C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_fid.png")
# plt.savefig("C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\sample\\complex\\cyclegan_fid.png")
