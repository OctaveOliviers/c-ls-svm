# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-17 16:59:17


# import libraries
from models.clssvm import *
import torch
import matplotlib.pyplot as plt
import tkinter


# def main():

# create patterns
x = torch.Tensor([[2], [7], [-7]])


# build model
model = CLSSVM()
model.add_layer(space='primal', dim_in=1, dim_out=4, feature_map='linear', p_err=1e2, p_drv=1e0, p_reg=1e-1)
model.add_layer(space='primal', dim_in=4, dim_out=3, feature_map='tanh', p_err=1e2, p_drv=1e1, p_reg=1e-1)
model.add_layer(space='dual', dim_in=3, dim_out=1, kernel='poly', kernel_param=2, p_err=1e2, p_drv=1e1, p_reg=1e-1)


# train model
cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
print("Training on", device)
# load data and model to device
x.to(device)
model.to(device)
model.custom_train(x, x, max_iter=100)


# visualize update equation
x_min, x_prec, x_max = -10, 0.1, 10
x_num = round(abs(x_max-x_min)/x_prec)
x_plot = torch.reshape(torch.linspace(x_min, x_max, x_num), (x_num,1))
f_plot = model.forward(x_plot)
# set visible axes
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
#  plot identity map
plt.plot(x_plot.detach().numpy(), x_plot.detach().numpy())
# plot update equation
plt.plot(x_plot.detach().numpy(), f_plot.detach().numpy())
# plot patterns to store
plt.plot(x.numpy(), x.numpy(), 'ro')
# show figure
plt.show()