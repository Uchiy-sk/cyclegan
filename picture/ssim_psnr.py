from PIL import Image
import numpy as np
import os

# 元となる画像の読み込み
filename_real = "C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\research\\picture\\cyclegan_disc.png"
img = Image.open(filename_real)
#オリジナル画像の幅と高さを取得
width, height = img.size

print(np.array(img))