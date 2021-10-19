# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-05-20 12:10:12
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-05-26 18:13:52

import torch
import matplotlib.pyplot as plt
import math

def plot_image(
        image:torch.Tensor,
        save:bool=False,
        name:str="test.png"
) -> None:
    num_pixels_side = int(math.sqrt(torch.numel(image)))
    plt.figure(figsize=(2,2))
    plt.imshow(image.detach().cpu().view((num_pixels_side, num_pixels_side)), cmap='gray')
    plt.axis("off")
    plt.savefig(name) if save else plt.show()
