# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:13:54
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-17 17:15:36


# import libraries
import torch
import torch.nn as nn
import gpytorch.kernels as gpk
from utils import *


class Layer(nn.Module):
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
        # 
        self._w_init     = getattr(nn.init, "xavier_uniform_")
        self._b_init     = getattr(nn.init, "zeros_")

    @property
    def p_err(self):
        """
        Getter for read-only attribute `_p_err`
        """
        return self._p_err
    

    @property
    def p_drv(self):
        """
        Getter for read-only attribute `_p_drv`
        """
        return self._p_drv
    

    @property
    def p_reg(self):
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
    def dim_in(self):
        """
        Getter for read-only attribute `_dim_in`
        """
        return self._dim_in


    @property
    def dim_out(self):
        """
        Getter for read-only attribute `_dim_out`
        """
        return self._dim_out


    @property
    def data(self):
        """
        Getter for read-only attribute `_data`
        """
        return self._data


    @data.setter
    def data(self, value):
        """
        Setter for read-only attribute `_data`
        """
        self._data = value
    

    def feat_jac_diag(self, x):
        """
        expain
        """
        # track the jacobian to x
        x.requires_grad_(True)
        # reset the gradient to zero
        x.grad = torch.zeros_like(x)
        # store diagonal of jacobian in x.grad
        for i in range(len(x)): self.feat_fun(x[i]).backward()
        return torch.squeeze(x.grad)


    def ker_grad(self, x, dy, create_graph=False):
        """
        explain
        """

        # grad = torch.empty(size=(len(x), len(dy), ))
        # # compute gradient dk(xp,yp) for each yp in dy
        # for xp in x:
        #     for yp in dy:


        # track the gradient to dy
        dy.requires_grad_(True)
        # reset the gradient to zero
        dy.grad = torch.zeros_like(dy)
        # compute kernel matrix
        # K = self.ker_fun(dy, x)
        print(self.ker_fun(dy, x))
        # return gradient
        # return torch.autograd.grad(K.chunk(len(K)), dy.chunk(len(dy)))[0]
        return torch.autograd.grad(self.ker_fun(dy, x), dy)[0]

        # return grad
        

    def ker_hess(self, dx, dy, create_graph=False):
        """
        explain
        """
        # track the gradient to dx and dy
        dx.requires_grad_(True)
        dy.requires_grad_(True)
        # reset the gradient to zero
        dx.grad = torch.zeros_like(dx)
        dy.grad = torch.zeros_like(dy)
        # return cross hessian
        return torch.autograd.functional.hessian(self.ker_fun, (dx,dy))[0][1]


    @staticmethod
    def build_feature_map(type, param):
        """
        explain
        """
        if type.lower() == "tanh":
            return torch.nn.Tanh()
        
        elif type.lower() == "linear":
            return torch.nn.Identity()
        
        elif type.lower() == "sign":
            return lambda x : torch.sign(x)

        elif type.lower() == "relu":
            return torch.nn.ReLu()

        elif type.lower() == "tanhshrink":
            return torch.nn.Tanhshrink()
        
        else:
            raise ValueError("Did not understand the name of the feature map.")


    @staticmethod
    def build_kernel(type, param):
        """
        x   bs1 x N
        y   bs2 x N
        out bs1 x bs2
        """
        if type.lower() == "rbf":
            # from https://jejjohnson.github.io/research_journal/snippets/pytorch/rbf_kernel/
            return lambda x, y : torch.squeeze(torch.exp(-.5*torch.sum((x[:,None,:]-y[None,:,:])**2,2)/param**2))
            # return 

        elif type.lower() == "poly":
            return lambda x, y : ((x@y.t()+1)**param).view(-1)
            # place torch.view on x and y
        
        else:
            raise ValueError("Did not understand the name of the kernel function.")


    @staticmethod
    def dimension_feature_space(dim_in, type):
        """
        explain
        """
        if type.lower() == "tanh":
            return dim_in

        elif type.lower() == "linear":
            return dim_in

        elif type.lower() == "sign":
            return dim_in

        elif type.lower() == "relu":
            return dim_in

        elif type.lower() == "tanhshrink":
            return dim_in

        else:
            raise ValueError("Did not understand the name of the feature map.")

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


    def __str__(self):
        """
        explain
        """
        return f"Primal layer with feature map '{self.feature_map} ({self.feature_param})'\n \
                 and hyper-parameters p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."


    @property
    def feature_map(self):
        """
        Getter for read-only attribute `_feature_map`
        """
        return self._feature_map
    

    @property
    def feature_param(self):
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

    
    def init_parameters(self, num_data):
        """
        explain
        """
        # store data
        # self.data = data
        # initialize layer parameters
        dim_feat = Layer.dimension_feature_space(self.dim_in, self.feature_map)
        self._weights = nn.Parameter(self._w_init(torch.Tensor(dim_feat, self.dim_out)), requires_grad=True)
        self._bias = nn.Parameter(self._b_init(torch.Tensor(self.dim_out,)), requires_grad=True)


    def forward(self, x, **kwargs):
        """
        explain
        """
        return torch.matmul(self.feat_fun(x), self.weights) + self.bias


    def loss(self, x, y):
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
        # self.ker_grad, self.ker_hess = Layer.build_kernel_grad_hess(self._kernel, self._kernel_param)
        # layer parameters
        self._weights_err = None
        self._weights_drv = None
        self._bias = None

        
    def __str__(self):
        """
        explain
        """
        return f"Dual layer with kernel '{self.kernel} ({self.kernel_param})'\n \
                 and hyper-parameters p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."


    @property
    def kernel(self):
        """
        Getter for read-only attribute `_kernel`
        """
        return self._kernel
    

    @property
    def kernel_param(self):
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

    
    def init_parameters(self, num_data):
        """
        explain
        """
        # initialize layer parameters
        self._weights_err = nn.Parameter(self._w_init(torch.Tensor(num_data, self.dim_out)), requires_grad=True)
        self._weights_drv = nn.Parameter(self._w_init(torch.Tensor(num_data, self.dim_out, self.dim_in)), requires_grad=True)
        self._bias = nn.Parameter(self._b_init(torch.Tensor(self.dim_out,)), requires_grad=True)


    def forward(self, x, **kwargs):
        """
        explain
        """
        print(x)

        # new output value
        x_new = torch.zeros_like(x)
        # sum over all data points
        for p, xp in enumerate(kwargs["base"]):
            # kernel term
            x_new += torch.outer(self.ker_fun(x, xp), self.weights_err[p,:])
            # kernel grad term
            x_new = self.weights_drv[p,:,:] @ self.ker_grad(xp, x)

        return x_new/self.p_reg + self.bias
        # return (self.weights_err@self.ker(self.data, x) + self.weights_drv@self.ker_grad(x, self.data))/self.p_reg + self.bias


    def loss(self, x, y):
        """
        explain
        """
 
        l_err = torch.norm(self.weights_err, p='fro')
        l_drv = torch.norm(self.weights_drv, p='fro')

        l_ker = 0
        for p, xp in enumerate(x):
            l_ker += self.ker_fun(xp, xp)*self.weights_err[p,:]@self.weights_err[p,:]
            for p_, xp_ in enumerate(x[:p,:]):
                l_ker += 2*self.ker_fun(xp, xp_)*self.weights_err[p,:]@self.weights_err[p_,:]

        l_hess = 0
        for p, xp in enumerate(x):
            l_hess += torch.trace(self.ker_hess(xp, xp)@self.weights_drv[p,:,:].t()@self.weights_drv[p,:,:])
            for p_, xp_ in enumerate(x[:p,:]):
                l_hess += 2*torch.trace(self.ker_hess(xp, xp_)@self.weights_drv[p,:,:].t()@self.weights_drv[p_,:,:])

        l_cross = 0
        for p, xp in enumerate(x):
            for p_, xp_ in enumerate(x):
                grad = self.ker_grad(xp_, xp)
                for i in range(self.dim_in):
                    l_cross += grad[i]*self.weights_err[p_,:]@self.weights_drv[p,:,i]
        
        l_y = torch.sum((y-self.bias)@self.weights_err.t())

        # we want to maximize this objective => return negative loss
        return l_err/2/self.p_err + l_drv/2/self.p_drv + (l_ker+l_hess+2*l_cross)/2/self.p_reg - l_y 

# end class LayerDual