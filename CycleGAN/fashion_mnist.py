import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import trans_holo

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_test = x_test[0] / 255
x_test = x_test.repeat(8, axis=0).repeat(8, axis=1)

h_test = trans_holo.asm(x_test, 5E-3, 532E-9)

img = [x_test, h_test, 255 - x_test, 255 - h_test]
title = ['mnist', 'holo', 'r_mnist', 'r_holo']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    plt.imshow(img[i])
    plt.gray()
    plt.axis('off')
plt.show()