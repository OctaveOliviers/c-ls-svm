# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:13:39
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 11:09:12


# TODO solve tied weigts (problem is dec_W remains on CPU)
# TODO

import torch
import torch.nn as nn
from tqdm import trange
import numbers
from datetime import datetime


class CAE(nn.Module):
    """
    docstring for CAE
    """

    def __init__(
            self,
            **kwargs
    ):
        """
        explain
        """
        super(CAE, self).__init__()

        dim_in = kwargs['dim_in']
        dim_hid = kwargs['dim_hid']
        # initialise encoder
        # self.encoder = nn.Linear(dim_in, dim_hid, bias=kwargs['bias'])
        # self.encoder.weight = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_hid, dim_in)), requires_grad=True)
        # if kwargs['bias']: self.encoder.bias = torch.nn.Parameter(nn.init.zeros_(torch.Tensor(dim_hid)), requires_grad=True)
        # initialise decoder
        # self.decoder = nn.Linear(dim_hid, dim_in, bias=kwargs['bias'])
        # self.decoder.weight = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_in, dim_hid)), requires_grad=True)
        # if kwargs['bias']: self.decoder.bias = torch.nn.Parameter(nn.init.zeros_(torch.Tensor(dim_in)), requires_grad=True)
        # if not kwargs['tied']: self.weight = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_in, dim_hid)), requires_grad=True)

        if kwargs.get('tied', True):
            self.enc_W = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_in, dim_hid)), requires_grad=True)
            self.dec_W = self.enc_W.t()
        else:
            self.enc_W = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_in, dim_hid)), requires_grad=True)
            self.dec_W = torch.nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_hid, dim_in)), requires_grad=True)

        self.enc_b = torch.nn.Parameter(nn.init.zeros_(torch.Tensor(dim_hid)), requires_grad=kwargs.get('bias', True))
        self.dec_b = torch.nn.Parameter(nn.init.zeros_(torch.Tensor(dim_in)), requires_grad=kwargs.get('bias', True))

        # activation function
        self.activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'linear':nn.Identity()}
        self.enc_act = self.activations[kwargs.get('encoder_activation', 'relu')]
        self.dec_act = self.activations[kwargs.get('decoder_activation', 'linear')]

        # regularisation hyper-parameter
        self.p_reg = kwargs.get('p_reg', 1e0)
        # initialise optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        # # training device
        # self._device = kwargs.get('device', 'cpu')

    def encoder_forward(self, x:torch.Tensor):
        return self.enc_act(x @ self.enc_W + self.enc_b)

    def forward(
            self,
            x:torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        explain
        """
        # hid = self.relu(self.encoder(x))
        hid = self.enc_act(x@self.enc_W + self.enc_b)
        # out = self.decoder(hid)
        out = self.dec_act(hid@self.dec_W + self.dec_b)

        return hid, out

    def loss(
            self,
            x:torch.Tensor
    ) -> numbers.Real:
        """
        explain
        """
        h, y = self.forward(x)
        # reconstruction error
        l_e = nn.MSELoss(reduction='sum')(x, y)
        # contraction
        l_c = sum([torch.norm(torch.autograd.functional.jacobian(self.encoder_forward, data), p='fro')**2 for data in x])
        # l_c = torch.norm(torch.autograd.functional.jacobian(self.encoder_forward, x), p='fro') ** 2
        # l_c = torch.norm(torch.autograd.functional.jacobian(self.forward, x)[0], p='fro') ** 2
        return l_e + self.p_reg*l_c

    def custom_train(
            self,
            x:torch.Tensor,
            max_iter:int=10**3,
            checkpoints:int=25,
            checkpointspath:str="../checkpoints/cae"
    ) -> None:
        """
        explain
        """
        print(f"First loss value = {self.loss(x)}")

        for _ in trange(max_iter):
            # try empty GPU
            # torch.cuda.empty_cache()
            self.opt.zero_grad()
            loss = self.loss(x)
            loss.backward()
            self.opt.step()

            # save checkpoint
            if _ % checkpoints == 0: self.save_model(epoch=_, loss=self.loss(x), path=checkpointspath)

        print(f"Final loss value = {self.loss(x)}")

    def save_model(
            self,
            epoch:int=None,
            loss:numbers.Real=None,
            path:str=None
    ) -> None:
        """
        explain
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': loss,
            }, path + f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}" + ".pt")

    def load_model(
            self,
            path:str=None
    ) -> None:
        """
        explain
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
