from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import trans_holo

z = -5E-3
filename = "C:\\Users\\uchiyama\\OneDrive - 千葉大学\\ドキュメント\\Python\\research\\picture\\cyclegan_holo\\600_100_100\\Epoch_100_1.jpg"

image = np.array(Image.open(filename))

size_x = image.shape[1]
size_y = image.shape[0]

# print(image[:, :, 0].shape)


rev_img = trans_holo.asm(image[:, :, 0], z)
# print(rev_img.shape)

plt.imshow(rev_img * 0.5 + 0.5)
plt.gray()
plt.show()