# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-31 18:13:52

from keras.datasets import mnist
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# seed the random generator
random.seed(42)

# load MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# select training images, remove padding, rescale between 0-1
num_img = 50
scale = 255
size_image = 20
train_idx = random.sample(range(len(train_X)), num_img)
train_img, train_lab = train_X[train_idx, 4:-4, 4:-4], train_y[train_idx]
train_img = np.divide(train_img, scale).reshape((num_img, size_image**2))

plt_i = random.randint(0,num_img)

# save original images to csv
pd.DataFrame(train_img).to_csv(f"../../data/MNIST/train-images-{num_img}.csv")
pd.DataFrame(train_lab).to_csv(f"../../data/MNIST/train-labels-{num_img}.csv")
plt.imshow(train_img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()

# inverse noise
img = 1.0 - train_img
pd.DataFrame(img).to_csv(f"../../data/MNIST/train-images-{num_img}-inverse.csv")
plt.imshow(img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()

# salt n pepa noise
img = train_img.reshape((num_img, size_image, size_image)).copy()
num_rand = int(0.5*img.size)
np.put(img, np.random.choice(img.size, size=num_rand, replace=False), np.random.rand(num_rand,))
img = img.reshape(num_img, 400)
pd.DataFrame(img).to_csv(f"../../data/MNIST/train-images-{num_img}-snp.csv")
plt.imshow(img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()

# small square noise
quarter = 5
img = train_img.reshape((num_img, size_image, size_image)).copy()
img[:, quarter:-quarter, quarter:-quarter] = np.random.rand(*img[:, quarter:-quarter, quarter:-quarter].shape)
img = img.reshape(num_img, 400)
pd.DataFrame(img).to_csv(f"../../data/MNIST/train-images-{num_img}-square.csv")
plt.imshow(img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()

# vertical half noise
img = train_img.reshape((num_img, size_image, size_image)).copy()
img[:, :, 0:10] = np.random.rand(*img[:, :, 0:10].shape)
img = img.reshape(num_img, 400)
pd.DataFrame(img).to_csv(f"../../data/MNIST/train-images-{num_img}-vertical.csv")
plt.imshow(img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()

# upper triangular half noise
img = train_img.reshape((num_img, size_image, size_image)).copy()
img = np.tril(img, k=-1) + np.triu(np.random.rand(*img.shape))
img = img.reshape(num_img, 400)
pd.DataFrame(img).to_csv(f"../../data/MNIST/train-images-{num_img}-diagonal.csv")
plt.imshow(img[plt_i,:].reshape((size_image, size_image)), cmap='gray')
plt.show()
