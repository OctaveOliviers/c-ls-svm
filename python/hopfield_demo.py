# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-03-31 15:26:59
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 11:09:05

# requires repo github.com/ml-jku/hopfield-layers saved in folder `hopfield_layers`

# import libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tkinter
from tqdm import trange

from hopfield_layers.modules import Hopfield, HopfieldLayer

# training on cpu or cuda
cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
print("Training on", device)

# create patterns
x = torch.Tensor([[2], [7], [-7], [1]]).to(device)

# build model
hopfield = Hopfield(input_size=1)
model = nn.Sequential(hopfield).to(device=device)
# training auxiliaries
optimiser = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='sum')
# train model
max_iter = 10**2
for epoch in trange(max_iter):
    optimiser.zero_grad()
    loss = criterion(model.forward(x), x)
    loss.backward()
    optimiser.step()

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