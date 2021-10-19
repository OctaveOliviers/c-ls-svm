# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-31 18:13:52

from modules.autoencoder import *
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# training on cpu or cuda
cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
print("Training on", device)

# create patterns
x = 0.5* torch.Tensor([[-15], [-8], [-7], [-1], [-2], [7], [15], [16]]).to(device)

# build model
dim_hid = 20
activation = "tanh"
model = CAE(dim_in=1, dim_hid=dim_hid, p_reg=10**1,
            encoder_activation=activation, decoder_activation='linear',
            bias=True, tied=False).to(device)

# train model
model.custom_train(x, max_iter=10**3)

# visualize update equation
x_min, x_prec, x_max = -10, 0.1, 10
x_num = round(abs(x_max-x_min)/x_prec)
x_plot = torch.reshape(torch.linspace(x_min, x_max, x_num).to(device), (x_num,1))
# update equation
h, f_plot = model.forward(x_plot)
# iterated update equation
f_conv = x_plot
for i in range(10**5):
    h, f_conv = model.forward(f_conv)
# set visible axes
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
#  plot identity map
plt.plot(x_plot.cpu().detach().numpy(), x_plot.cpu().detach().numpy())
# plot update equation
plt.plot(x_plot.cpu().detach().numpy(), f_plot.cpu().detach().numpy())
plt.plot(x_plot.cpu().detach().numpy(), f_conv.cpu().detach().numpy())
# plot patterns to store
plt.plot(x.cpu().detach().numpy(), x.cpu().detach().numpy(), 'ro')
# show figure
plt.show()

# # store data
# df = pd.DataFrame(np.concatenate((f_plot.cpu().detach().numpy(),
#                                   f_conv.cpu().detach().numpy()),
#                                  axis=1))
# df.to_csv(f"../../data/cae_1d_{activation}.csv")
