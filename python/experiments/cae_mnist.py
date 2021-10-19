# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-31 18:13:52

# from keras.datasets import mnist
from modules.autoencoder import *
from utils.plot_image import *
import torch
import math, random
import pandas as pd

# seed the random generator
random.seed(42)
FIGS = "../../figs/cls-cae/mnist/"
DATA = "../../data/MNIST/"

# training on cpu or cuda
cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
print(f"Training on {device}")

csv_to_torch = lambda str : torch.tensor(pd.read_csv(str).values).float().to(device)[:,1::]

# load MNIST dataset
train_img = csv_to_torch(f"{DATA}train-images-50-original.csv")
num_img, dim_img = train_img.shape
size_img = math.sqrt(dim_img)

# build model
model = CAE(dim_in=dim_img, dim_hid=400, p_reg=1e-4,
            encoder_activation='relu', decoder_activation='sigmoid',
            bias=False, tied=False).to(device)

# train model
model.custom_train(train_img, max_iter=10**2)
# load model form checkpoint
# model.load_model('../checkpoints/cae21-05-26-11-54-07.pt')

# evaluate model
max_retrieval_iter = 201
plot_step = [0,1,10,100,max_retrieval_iter-1]
# compare effect of different types of noise
test_idx = random.randint(0,num_img) # 40
noisy_sets = ["original", "snp", "diagonal", "inverse", "square", "vertical"]
for set in noisy_sets:
    train_img_noisy = csv_to_torch(f"{DATA}train-images-50-{set}.csv")
    test_img = train_img_noisy[test_idx, :]
    for i in range(max_retrieval_iter):
        if i in plot_step: plot_image(test_img, save=True, name=f"{FIGS}cae-{set}-{i}.png")
        h, test_img = model.forward(test_img)
# choose 5 random snp noised images
test_indices = random.sample(range(num_img), 5) # [7, 1, 47, 17, 15]
train_img_noisy = csv_to_torch(f"{DATA}train-images-50-snp.csv")
for j, idx in enumerate(test_indices):
    test_img = train_img_noisy[idx,:]
    plot_image(train_img[idx,:], save=True, name=f"{FIGS}cae-{j+1}-original.png")
    for i in range(max_retrieval_iter):
        if i in plot_step: plot_image(test_img, save=True, name=f"{FIGS}cae-{j+1}-{i}.png")
        h, test_img = model.forward(test_img)
