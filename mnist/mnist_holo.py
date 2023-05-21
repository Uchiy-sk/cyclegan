from keras.utils import np_utils  # エラー対策？
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from trans_holo import asm
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import gc

# MNISTをDL
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_train = 60
num_test = 10
x_train = x_train[:num_train]
x_test = x_test[:num_test]
y_train = y_train[:num_train]
y_test = y_test[:num_test]
# print(x_train.shape)
# print(x_test.shape)


# 画像出力関数
def imprint(f):
    fig = plt.figure(figsize=(20, 2))
    n = 10
    for i in range(10):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(f[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show

# %%
# 画像拡大関数


def expand(f, ex, ax):
    f_ex = f.repeat(ex, axis=ax)
    return f_ex


# %%
ex = 8
x_train = expand(expand(x_train, ex, 1), ex, 2)
x_test = expand(expand(x_test, ex, 1), ex, 2)
# print(x_train.shape)
# print(x_test.shape)


z = 0.05
g_train = np.zeros_like(x_train)
g_test = np.zeros_like(x_test)

for i in range(num_train):
    g_train[i] = asm(x_train[i], z, 532E-9)[0]

for j in range(num_test):
    g_test[j] = asm(x_test[j], z, 532E-9)[0]


# 機械学習
image_x, image_y = 28*ex, 28*ex  # 画像サイズ
batch_size = 128  # バッチサイズ
input_shape = (image_x, image_y, 1)
num = 10  # 0~9番号
epochs = 10


m_train = g_train.reshape(g_train.shape[0], image_x, image_y, 1)
m_test = g_test.reshape(g_test.shape[0], image_x, image_y, 1)

m_train = m_train.astype('float32') / 255
m_test = m_test.astype('float32') / 255


# One-hotベクトル変換
y_train = np_utils.to_categorical(y_train, num)
y_test = np_utils.to_categorical(y_test, num)


# モデル作成
model = Sequential()
model.add(Conv2D(32,  # filter:フィルタの個数
                 kernel_size=3,  # 2次元(3,3)畳み込みウィンドウの幅と高さ
                 activation='relu',
                 input_shape=input_shape))  # 入力shape
model.add(MaxPooling2D(pool_size=(2, 2)))  # 領域内で最大値をとるモデル
model.add(Conv2D(64, (3, 3), activation='relu'))  # relu:正規化線形関数
model.add(Dropout(0.25))
model.add(Flatten())  # 特徴マップをベクトル形式に変換
model.add(Dense(num, activation='softmax'))  # softmax:0~1に確率変換
model.summary()

# 損失関数・最適化関数、評価指数の指定とコンパイル
model.compile(loss='categorical_crossentropy',  # 損失関数 -> https://keras.io/objectives/
              optimizer='Adam',  # 評価関数 -> https://keras.io/optimizer/
              metrics=['accuracy'])


# モデル学習
hist = model.fit(m_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(m_test, y_test))
# validation_split = 0.2) #検証用データの割合


# モデル評価
score = model.evaluate(m_test, y_test)
print('Test loss:', score[0])  # 損失
print('Test acc :', score[1])  # 正解率


# グラフ化
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(1, 2, 1, xlabel="epoch", ylabel="accuracy")
ax2 = fig.add_subplot(1, 2, 2, xlabel="epoch", ylabel="loss")

acc = hist.history['accuracy']
loss = hist.history['loss']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']

ax1.plot(range(len(acc)), acc, marker='o', label='accuracy')
ax1.plot(range(len(val_acc)), val_acc, marker='^', label='val_accuracy')
ax1.legend(loc='best', fontsize=10)
ax1.grid()
# ax1.xlabel('epoch')
# ax1.ylabel('accuracy')

ax2.plot(range(len(loss)), loss, marker='.', label='loss')
ax2.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss')
ax2.legend(loc='best', fontsize=10)
ax2.grid()
# ax2.xlabel('epoch')
# ax2.ylabel('loss')

plt.tight_layout()
plt.show()

fig.savefig("mnist_holo.png")
