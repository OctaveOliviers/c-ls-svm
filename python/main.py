# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-31 18:13:52


# import libraries
from modules.clssvm import *
import torch
import matplotlib.pyplot as plt
import tkinter

# training on cpu or cuda
cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
print("Training on", device)

# create patterns
x = torch.Tensor([[2], [7], [-7], [1]]).to(device)

# build model
model = CLSSVM()
# model.add_layer(space='primal', dim_in=1, dim_out=3, feature_map='linear',          p_err=1e2, p_drv=1e0, p_reg=1e-2)
# model.add_layer(space='primal', dim_in=3, dim_out=1, feature_map='tanh',            p_err=1e2, p_drv=1e0, p_reg=1e-2)
model.add_layer(space='dual',   dim_in=1, dim_out=1, kernel='rbf', kernel_param=5, p_err=1e2, p_drv=1e2, p_reg=1e-2)
model.to(device)

# train model
model.custom_train(x, x, max_iter=10**3)

# time functions to find bottleneck

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