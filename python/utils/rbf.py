# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2021-01-28 12:14:57
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-28 22:31:11


class ExponentialKernel(nn.Module):
    def __init__(self, size_in, num_kernels, sigma, sigma_trainable=False):
        # size_in: size of x
        # size_out: number of support vectors (alpha_i), number of neurons
        super(ExponentialKernel, self).__init__()
        self.size_in = size_in
        self.num_kernels = num_kernels
        self.sigma = sigma

        # too close to center, take random points from different classes
        self.param = nn.Parameter(4*nn.init.orthogonal_(torch.Tensor(self.size_in, self.num_kernels)))

    def forward(self, x, idx_sv=None):
        num_points = x.size(0)

        xs = x[:, :, None].expand(-1, -1, self.num_kernels)
        params = self.param.expand(num_points, -1, -1)

        diff = xs - params
        norm2 = torch.sum(diff * diff, axis=1)
        fact = 1/(2*self.sigma_trainable**2)
        output = torch.exp(-fact * norm2)

        return output

    def special_init(self, x):
        self.param = nn.Parameter(x.clone().t())