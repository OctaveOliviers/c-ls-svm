# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:13:54
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-31 18:24:42


import torch
import torch.nn as nn
from abc import ABCMeta
import numbers


class Layer(nn.Module, metaclass=ABCMeta):
    """
    docstring for Layer
    """

    def __init__(self, **kwargs):
        """ 
        explain
        """
        super(Layer, self).__init__()

        # hyper-parameters
        self._p_err      = kwargs['p_err']        
        self._p_drv      = kwargs['p_drv']
        self._p_reg      = kwargs['p_reg']
        # self.p_mom       = kwargs['p_mom']
        # layer parameters
        self._criterion  = nn.MSELoss(reduction='sum')
        self._dim_in     = kwargs['dim_in']
        self._dim_out    = kwargs['dim_out']
        # data to train on
        self._data       = None
        # parameter initialisation functions
        self._w_init     = getattr(nn.init, "xavier_uniform_")
        self._b_init     = getattr(nn.init, "zeros_")

    @property
    def p_err(self) -> numbers.Real:
        """
        Getter for read-only attribute `_p_err`
        """
        return self._p_err

    @property
    def p_drv(self) -> numbers.Real:
        """
        Getter for read-only attribute `_p_drv`
        """
        return self._p_drv

    @property
    def p_reg(self) -> numbers.Real:
        """
        Getter for read-only attribute `_p_reg`
        """
        return self._p_reg

    @property
    def criterion(self):
        """
        Getter for read-only attribute `_criterion`
        """
        return self._criterion

    @property
    def dim_in(self) -> numbers.Real:
        """
        Getter for read-only attribute `_dim_in`
        """
        return self._dim_in

    @property
    def dim_out(self) -> numbers.Real:
        """
        Getter for read-only attribute `_dim_out`
        """
        return self._dim_out

    @property
    def data(self) -> torch.Tensor:
        """
        Getter for read-only attribute `_data`
        """
        return self._data

    @data.setter
    def data(
            self,
            value:torch.Tensor
    ) -> None:
        """
        Setter for read-only attribute `_data`
        """
        self._data = value

    def feat_jac_diag(
            self,
            x:torch.Tensor
    ) -> torch.Tensor:
        """
        explain
        """
        # track the jacobian to x
        x.requires_grad_(True)
        # reset the gradient to zero
        x.grad = torch.zeros_like(x)
        # store diagonal of jacobian in x.grad
        self.feat_fun(x).backward(torch.ones_like(x))
        return torch.squeeze(x.grad)

    def ker_grad(
            self,
            x:torch.Tensor,
            dy:torch.Tensor
    ) -> torch.Tensor:
        """
        explain
        """
        # track the gradient to dy
        dy.requires_grad_(True)
        # reset the gradient to zero
        dy.grad = torch.zeros_like(dy)
        # retain gradient wrt second input
        grad = torch.squeeze(torch.autograd.functional.jacobian(self.ker_fun, (x, dy))[1])
        if self.dim_in == 1: 
            return grad.unsqueeze(-1)
        else:
            return grad

    def ker_hess(
            self,
            dx:torch.Tensor,
            dy:torch.Tensor
    ) -> torch.Tensor:
        """
        explain
        """
        # track the gradient to dx and dy
        dx.requires_grad_(True)
        dy.requires_grad_(True)
        # # reset the gradient to zero
        dx.grad = torch.zeros_like(dx)
        dy.grad = torch.zeros_like(dy)
        # return cross hessian
        return torch.autograd.functional.hessian(self.ker_fun, (dx,dy))[0][1]

    @staticmethod
    def build_feature_map(
            name:str,
            param:numbers.Real
    ):
        """
        explain
        """
        if name.lower() == "tanh":
            return nn.Tanh()
        
        elif name.lower() == "linear":
            return nn.Identity()
        
        elif name.lower() == "sign":
            return lambda x : torch.sign(x)

        elif name.lower() == "relu":
            return nn.ReLU()

        elif name.lower() == "tanhshrink":
            return nn.Tanhshrink()
        
        else:
            raise ValueError(f"Did not understand the name of the feature map: {name}.")

    @staticmethod
    def build_kernel(
            name:str,
            param:numbers.Real
    ):
        """
        x   bs1 x N
        y   bs2 x N
        out bs1 x bs2
        """
        if name.lower() == "rbf":
                        
            def rbf_ker(x, y):
                # from https://jejjohnson.github.io/research_journal/snippets/pytorch/rbf_kernel/
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                if len(y.shape) == 1:
                    y = y.unsqueeze(0)
                return torch.exp(-.5*torch.sum((x[:,None,:]-y[None,:,:])**2,2)/param**2)

            return rbf_ker

        elif name.lower() == "poly":

            def poly_ker(x, y):
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                if len(y.shape) == 1:
                    y = y.unsqueeze(0)
                return torch.pow(x@y.t()+1, param)

            return poly_ker
        
        else:
            raise ValueError(f"Did not understand the name of the kernel function: {name}.")

    @staticmethod
    def dimension_feature_space(
            dim_in:numbers.Real,
            name:str
    ) -> numbers.Real:
        """
        explain
        """
        if name.lower() == "tanh":
            return dim_in

        elif name.lower() == "linear":
            return dim_in

        elif name.lower() == "sign":
            return dim_in

        elif name.lower() == "relu":
            return dim_in

        elif name.lower() == "tanhshrink":
            return dim_in

        else:
            raise ValueError(f"Did not understand the name of the feature map: {name}.")

# end class Layer


class LayerPrimal(Layer):
    """
    docstring for LayerPrimal
    """

    def __init__(self, **kwargs):
        """
        explain
        """
        super(LayerPrimal, self).__init__(**kwargs)

        # layer feature map
        self._feature_map   = kwargs.get('feature_map', 'linear')
        self._feature_param = kwargs.get('feature_param', None)
        self.feat_fun = Layer.build_feature_map(self._feature_map, self._feature_param)
        # self.map_jac, self.map_jac_diag = Layer.build_feature_jacobian(self._feat_map, self._feat_param)
        # layer parameters
        self._weights = None
        self._bias = None

    def __str__(self) -> str:
        """
        explain
        """
        return f"Primal layer with feature map '{self.feature_map} ({self.feature_param})'\n \
                 and hyper-parameters p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."

    @property
    def feature_map(self) -> str:
        """
        Getter for read-only attribute `_feature_map`
        """
        return self._feature_map

    @property
    def feature_param(self) -> numbers.Real:
        """
        Getter for read-only attribute `_feature_param`
        """
        return self._feature_param

    @property
    def weights(self):
        """
        Getter for read-only attribute `_weights`
        """
        return self._weights

    @property
    def bias(self):
        """
        Getter for read-only attribute `_bias`
        """
        return self._bias
    
    def init_parameters(
            self,
            num_data:int,
            device='cpu'
    ) -> None:
        """
        explain
        """
        # initialize layer parameters
        dim_feat = Layer.dimension_feature_space(self.dim_in, self.feature_map)
        self._weights = nn.Parameter(self._w_init(torch.Tensor(dim_feat, self.dim_out)).to(device), requires_grad=True)
        self._bias = nn.Parameter(self._b_init(torch.Tensor(self.dim_out,)).to(device), requires_grad=True)

    def forward(
            self,
            x:torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        explain
        """
        return torch.matmul(self.feat_fun(x), self.weights) + self.bias

    def loss(
            self,
            x:torch.Tensor,
            y:torch.Tensor
    ) -> numbers.Real:
        """
        explain
        """
        # equilibrium objective
        e = self.criterion(self.forward(x), y)
        # local stability objective
        s = sum([torch.norm(self.weights.t()*j_diag, p='fro') for j_diag in self.feat_jac_diag(x)])
        # regularisation objective
        r = torch.norm(self.weights, p='fro')
        # return weighted sum of objectives 
        return (self.p_err*e + self.p_drv*s + self.p_reg*r)/2.

# end class LayerPrimal


class LayerDual(Layer):
    """
    docstring for LayerDual
    """

    def __init__(self, **kwargs):
        """
        explain
        """
        super(LayerDual, self).__init__(**kwargs)

        # layer kernel
        self._kernel = kwargs.get('kernel', 'rbf')
        self._kernel_param = kwargs.get('kernel_param', 1.)
        self.ker_fun = Layer.build_kernel(self._kernel, self._kernel_param)
        # layer parameters
        self._weights_err = None
        self._weights_drv = None
        self._bias = None

    def __str__(self) -> str:
        """
        explain
        """
        return f"Dual layer with kernel '{self.kernel} ({self.kernel_param})'\n \
                 and hyper-parameters p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."

    @property
    def kernel(self) -> str:
        """
        Getter for read-only attribute `_kernel`
        """
        return self._kernel

    @property
    def kernel_param(self) -> numbers.Real:
        """
        Getter for read-only attribute `_kernel_param`
        """
        return self._kernel_param

    @property
    def weights_err(self):
        """
        Getter for read-only attribute `_weights_err`
        """
        return self._weights_err

    @property
    def weights_drv(self):
        """
        Getter for read-only attribute `_weights_drv`
        """
        return self._weights_drv

    @property
    def bias(self):
        """
        Getter for read-only attribute `_bias`
        """
        return self._bias
    
    def init_parameters(
            self,
            num_data:int,
            device:str='cpu'
    ) -> None:
        """
        explain
        """
        # initialize layer parameters
        self._weights_err = nn.Parameter(self._w_init(torch.Tensor(num_data, self.dim_out)).to(device), requires_grad=True)
        self._weights_drv = nn.Parameter(self._w_init(torch.Tensor(num_data, self.dim_in, self.dim_out)).to(device), requires_grad=True)
        self._bias = nn.Parameter(self._b_init(torch.Tensor(self.dim_out,)).to(device), requires_grad=True)

    def forward(
            self,
            x:torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        explain
        """
        # new output value
        x_new = torch.zeros_like(x)
        # kernel term
        x_new += self.ker_fun(x, kwargs["targets"]) @ self.weights_err
        # kernel grad term
        for p, xp in enumerate(kwargs["targets"]):
            x_new += self.ker_grad(x, xp) @ self.weights_drv[p, :, :]

        return x_new/self.p_reg + self.bias

    def loss(
            self,
            x:torch.Tensor,
            y:torch.Tensor
    ) -> numbers.Real:
        """
        explain
        """

        ##################################
        # # regularisation
        # l_err = torch.norm(self.weights_err, p='fro')**2 / self.p_err
        # l_drv = torch.norm(self.weights_drv, p='fro')**2 / self.p_drv
        #
        # # kernel term
        # l_ker = torch.norm(self.ker_fun(x, x) @ self.weights_err, p='fro')**2 / self.p_reg
        #
        # # kernel hessian term
        # mat_hess = torch.zeros_like(self.weights_drv)
        # for p, xp in enumerate(x):
        #     mat_hess[p, :, :] += self.ker_hess(xp, xp) @ self.weights_drv[p, :, :]
        #     for p_, xp_ in enumerate(x[:p, :]):
        #         hess = self.ker_hess(xp, xp_)
        #         mat_hess[p, :, :] += hess @ self.weights_drv[p_, :, :]
        #         mat_hess[p_, :, :] += hess @ self.weights_drv[p, :, :]
        # l_hess = torch.norm(mat_hess, p='fro')**2 / self.p_reg
        #
        # # kernel cross term
        # mat_cross_err = torch.zeros_like(self.weights_drv)
        # mat_cross_drv = torch.zeros_like(self.weights_err)
        # for p, xp in enumerate(x):
        #     grad = self.ker_grad(x, xp)
        #     mat_cross_err[p, :, :] = grad.t() @ self.weights_err
        #     mat_cross_drv += grad @ self.weights_drv[p, :, :]
        # l_cross = (torch.norm(mat_cross_err, p='fro')**2 + torch.norm(mat_cross_drv, p='fro')**2) / self.p_reg
        #
        # l_y = torch.norm(y - self.bias, p='fro')**2
        #
        # # we want to maximize this objective => return negative loss
        # return torch.squeeze(l_err + l_drv + l_ker + l_hess + l_cross - l_y)

        ##################################
        # regularisation
        l_err = torch.norm(self.weights_err, p='fro') ** 2 / 2 / self.p_err
        l_drv = torch.norm(self.weights_drv, p='fro') ** 2 / 2 / self.p_drv

        # kernel term
        l_ker = 0.
        for p, xp in enumerate(x):
            l_ker += self.ker_fun(xp, xp) * self.weights_err[p, :] @ self.weights_err[p, :]
            for p_, xp_ in enumerate(x[:p, :]):
                l_ker += 2 * self.ker_fun(xp, xp_) * self.weights_err[p, :] @ self.weights_err[p_, :]

        # kernel hessian term
        l_hess = 0.
        for p, xp in enumerate(x):
            # l_hess += torch.trace(self.ker_hess(xp, xp) @ self.weights_drv[p,:,:].t() @ self.weights_drv[p,:,:])
            l_hess += torch.sum(self.ker_hess(xp, xp) * (self.weights_drv[p, :, :] @ self.weights_drv[p, :, :].t()))
            for p_, xp_ in enumerate(x[:p, :]):
                # l_hess += 2 * torch.trace(self.ker_hess(xp, xp_) @ self.weights_drv[p, :, :].t() @ self.weights_drv[p_, :, :])
                l_hess += 2 * torch.sum(self.ker_hess(xp, xp_) * (self.weights_drv[p, :, :] @ self.weights_drv[p_, :, :].t()))

        # kernel cross term
        l_cross = 0.
        for p, xp in enumerate(x):
            for p_, xp_ in enumerate(x):
                grad = self.ker_grad(xp_, xp)
                # for i in range(self.dim_in):
                #     l_cross += grad[i]*self.weights_err[p_,:]@self.weights_drv[p,:,i]
                l_cross += torch.sum(self.weights_err[p_, :] @ self.weights_drv[p, :, :].t() @ grad)

        l_y = torch.sum((y - self.bias) @ self.weights_err.t())

        # we want to maximize this objective => return negative loss
        return -torch.squeeze(l_err + l_drv + (l_ker + l_hess + 2 * l_cross) / 2 / self.p_reg - l_y)

# end class LayerDual
