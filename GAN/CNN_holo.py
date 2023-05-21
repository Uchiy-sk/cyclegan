#%%
import cv2
from PIL import Image
from cv2 import sort
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.client import device_lib

from trans_holo import asm

ex = 8
z = 3E-2        # 伝搬距離
num_train = 60
num_test = 10
batch_size = 64 #バッチサイズ
image_x, image_y = 28*ex, 28*ex
input_shape = (image_x, image_y, 1)
epochs = 50
path = 'C:\\Users\\Uchiy\\OneDrive - 千葉大学\\ドキュメント\\Python\\research'

print('start:trans_holo...')
h_train = np.zeros((num_train, image_x, image_y), dtype=np.complex64)
h_test = np.zeros((num_test, image_x, image_y), dtype=np.complex64)

# 画像読み込み
# for i in range(num_train):
#     h_train[i] = np.array(Image.open(path + f'\\mnist_data\\train\\mnist_train_{i}.png').convert('L'))
# print('Load Completed : h_train')

# for j in range(num_test):
#     h_test[j] = np.array(Image.open(path + f'\\mnist_data\\test\\mnist_test_{j}.png').convert('L'))
# print('Load Completed : h_test','\n')
# print(h_train.shape)

# 画像拡大
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[:num_train]
x_test = x_test[:num_test]
x_train = x_train.repeat(ex, axis=1).repeat(ex, axis=2)
x_test = x_test.repeat(ex, axis=1).repeat(ex, axis=2)
# print(x_train.shape, '\n')

# ホログラム変換
for i in range(num_train):
    h_train[i] = asm(x_train[i], z)
for j in range(num_test):
    h_test[j] = asm(x_test[j], z)
    
plt.imshow(x_train[0].real)
plt.gray()
plt.axis('off')
plt.show()

plt.imshow(h_train[0].real)
plt.gray()
plt.axis('off')
plt.show()

plt.imshow(asm(h_train[0], -z).real)
plt.gray()
plt.axis('off')
plt.show()
# print('completed:trans_holo.\n\n')

# 機械学習
print("Tensorflow-GPU Version :", tf.__version__)
print("Keras Version :", keras.__version__)
device_lib.list_local_devices()

# 前処理
h_train = h_train.reshape(h_train.shape[0], h_train.shape[1], h_train.shape[2], 1)
h_test = h_test.reshape(h_test.shape[0], h_test.shape[1], h_test.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

h_train = h_train.real.astype('complex64') / 255
h_test = h_test.real.astype('complex64') / 255
x_train = x_train.astype('complex64') / 255
x_test = x_test.astype('complex64') / 255

print(h_train.max(), '\n',h_train.min())
print(x_train.max(), '\n',x_train.min())

#%%
# モデル作成
model = Sequential()
model.add(Conv2D(32,                            #filter:フィルタの個数
                kernel_size = 3,                #2次元(3,3)畳み込みウィンドウの幅と高さ
                activation = 'relu',
                padding = 'same',
                input_shape = input_shape))     #入力shape
model.add(MaxPooling2D(pool_size =(2, 2)))      #領域内で最大値をとるモデル
model.add(Conv2D(64, (3,3),
                activation = 'relu',
                padding = 'same'))              #relu:正規化線形関数
model.add(UpSampling2D((2, 2))) 
# model.add(Dropout(0.25))
model.add(Conv2D(1, (3,3),
                activation = 'relu',
                padding = 'same'))
model.summary()

#損失関数・最適化関数、評価指数の指定とコンパイル
model.compile(loss = 'binary_crossentropy', #損失関数 -> https://keras.io/objectives/
              optimizer = 'Adam', #評価関数 -> https://keras.io/optimizer/
              metrics = ['accuracy'])


# print('x_train : ', x_train.shape)
# print('h_train : ', h_train.shape)

# 学習
print('Now Learning...')
hist = model.fit(x_train, h_train,
                batch_size = batch_size,
                epochs = epochs,
                shuffle = True,
                verbose = 1,
                validation_data = (x_test, h_test)) #検証用データ


#モデルの評価
score = model.evaluate(x_test, h_test)
print('Test loss:', score[0]) #損失
print('Test acc :', score[1], '\n') #正解率


# 予測画像出力
print('image predict:')
n = 10
result = model.predict(x_test)
result = result.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
result *= 255

fig = plt.figure(figsize=(10, 3))
for i in range(n):
    ax1 = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].real.reshape(image_x, image_y))
    plt.gray()
    ax1.axis('off')
    
    ax2 = plt.subplot(3, n, n+i+1)
    plt.imshow(result[i].real.reshape(image_x, image_y))
    plt.gray()
    ax2.axis('off')
    
    ax3 = plt.subplot(3, n, 2*n+i+1)
    plt.imshow(asm(result[i], -z).real.reshape(image_x, image_y))
    plt.gray()
    ax3.axis('off')
plt.show()

# fig.savefig("mnist_holo.png")

fig.savefig(path + "\\picture\\holo_predict.png")


# %%
