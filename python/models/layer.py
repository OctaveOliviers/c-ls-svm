# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-01-28 12:13:54
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-03 18:52:29


# import libraries
import torch
import torch.nn as nn
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

    @property
    def p_err(self):
        """
        Getter for read-only attribute '_p_err'
        """
        return self._p_err
    

    @property
    def p_drv(self):
        """
        Getter for read-only attribute '_p_drv'
        """
        return self._p_drv
    

    @property
    def p_reg(self):
        """
        Getter for read-only attribute '_p_reg'
        """
        return self._p_reg
    

    @property
    def criterion(self):
        """
        Getter for read-only attribute '_criterion'
        """
        return self._criterion
    

    @property
    def dim_in(self):
        """
        Getter for read-only attribute '_dim_in'
        """
        return self._dim_in


    @property
    def dim_out(self):
        """
        Getter for read-only attribute '_dim_out'
        """
        return self._dim_out


    @property
    def data(self):
        """
        Getter for read-only attribute '_data'
        """
        return self._data


    @data.setter
    def data(self, value):
        """
        Setter for read-only attribute '_data'
        """
        self._data = value
    

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
    def build_feature_jacobian(type, param):
        """
        explain
        """
        if type.lower() == "tanh":
            return lambda x : torch.cat([torch.diag(1./torch.cosh(x[p,:])**2) for p in range(x.shape[0])], dim=1), \
                   lambda x : 1./torch.cosh(x)**2

        elif type.lower() == "linear":
            return lambda x : torch.eye(x.shape[1]).repeat(1,x.shape[0]), \
                   lambda x : torch.ones_like(x)

        elif type.lower() == "sign":
            return lambda x : torch.zeros(x.shape[1], torch.numel(x)), \
                   lambda x : torch.zeros_like(x)

        elif type.lower() == "relu":
            return lambda x : torch.cat([torch.diag(torch.round(torch.sigmoid(x[p,:]))) for p in range(x.shape[0])], dim=1), \
                   lambda x : torch.round(torch.sigmoid(x))

        elif type.lower() == "tanhshrink":
            return lambda x : torch.cat([torch.diag(torch.tanh(x[p,:])**2) for p in range(x.shape[0])], dim=1), \
                   lambda x : torch.tanh(x)**2

        else:
            raise ValueError("Did not understand the name of the feature map.")


    @staticmethod
    def build_kernel(type, param):
        """
        explain
        """
        if type.lower() == "rbf":
            # from https://jejjohnson.github.io/research_journal/snippets/pytorch/rbf_kernel/
            return lambda x, y : torch.exp(-.5*torch.sum((x[:,None,:]-y[None,:,:])**2,2)/param**2)
        
        elif type.lower() == "poly":
            return lambda x, y : (x@y.t()+1)**param
        
        else:
            raise ValueError("Did not understand the name of the kernel function.")


    @staticmethod
    def build_kernel_grad_hess(type, param):
        """
        explain
        """
        if type.lower() == "rbf":
            return lambda x, dy : Layer.build_kernel(type, param)(x, dy)*(x-dy)/param**2, \
                   lambda dx, dy : 

        elif type.lower() == "poly":
            return lambda x, dy : param*(x@dy.t()+1)**(param-1)*x, \
                   lambda x, dy : 

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
        self._feat_map   = kwargs.get('feat_map', 'linear')
        self._feat_param = kwargs.get('feat_param', None)
        self.map = Layer.build_feature_map(self._feat_map, self._feat_param)
        self.map_jac, self.map_jac_diag = Layer.build_feature_jacobian(self._feat_map, self._feat_param)
        # layer parameters
        self._weights = None
        self._bias = None


    def __str__(self):
        """
        explain
        """
        return f"Primal layer with feature map '{self.feat_map} ({self.feat_param})' and p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."


    @property
    def feat_map(self):
        """
        Getter for read-only attribute '_feat_map'
        """
        return self._feat_map
    

    @property
    def feat_param(self):
        """
        Getter for read-only attribute '_feat_param'
        """
        return self._feat_param


    @property
    def weights(self):
        """
        Getter for read-only attribute '_weights'
        """
        return self._weights


    @property
    def bias(self):
        """
        Getter for read-only attribute '_bias'
        """
        return self._bias

    
    def init_parameters(self, data):
        """
        explain
        """
        # store data
        self.data = data
        # initialize layer parameters
        dim_feat = Layer.dimension_feature_space(self.dim_in, self.feat_map)
        self._weights = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(dim_feat, self.dim_out)), requires_grad=True)
        self._bias = nn.Parameter(nn.init.zeros_(torch.Tensor(self.dim_out,)), requires_grad=True)


    def forward(self, x):
        """
        explain
        """
        return torch.matmul(self.map(x), self.weights) + self.bias


    def loss(self, x, y):
        """
        explain
        """
        # equilibrium objective
        e = self.criterion(self.forward(x), y)
        # local stability objective
        s = sum([torch.norm(self.weights.t()*j_diag, p='fro') for j_diag in self.map_jac_diag(x)])
        # regularisation objective
        r = torch.norm(self.weights, p='fro')
        # return weighted sum of objectives 
        return self.p_err*e + self.p_drv*s + self.p_reg*r

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
        self.ker = Layer.build_kernel(self._kernel, self._kernel_param)
        self.ker_grad, self.ker_hess = Layer.build_kernel_grad_hess(self._kernel, self._kernel_param)
        # layer parameters
        self._weights_err = None
        self._weights_drv = None
        self._bias = None

        
    def __str__(self):
        """
        explain
        """
        return f"Dual layer with kernel '{self.kernel} ({self.kernel_param})' and p_err={self.p_err}, p_drv={self.p_drv}, p_reg={self.p_reg}."


    @property
    def kernel(self):
        """
        Getter for read-only attribute '_kernel'
        """
        return self._kernel
    

    @property
    def kernel_param(self):
        """
        Getter for read-only attribute '_kernel_param'
        """
        return self._kernel_param


    @property
    def weights_err(self):
        """
        Getter for read-only attribute '_weights_err'
        """
        return self._weights_err
    

    @property
    def weights_drv(self):
        """
        Getter for read-only attribute '_weights_drv'
        """
        return self._weights_drv


    @property
    def bias(self):
        """
        Getter for read-only attribute '_bias'
        """
        return self._bias

    
    def init_parameters(self, data):
        """
        explain
        """
        # store data
        self.data = data
        # initialize layer parameters
        self._weights_err = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.dim_out, data.shape[0])), requires_grad=True)
        self._weights_drv = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.dim_out, self.dim_in*data.shape[0])), requires_grad=True)
        self._bias = nn.Parameter(nn.init.zeros_(torch.Tensor(self.dim_out,)), requires_grad=True)


    def forward(self, x):
        """
        explain
        """
        return (self.weights_err@self.ker(self.data, x) + self.weights_drv@self.ker_grad(x, self.data))/self.p_reg + self.bias


    def loss(self, x, y):
        """
        explain
        """
        # # equilibrium objective
        # e = self.criterion(self.forward(x), y)
        # # local stability objective
        # s = sum([torch.norm(self.weights.t()*j_diag, p='fro') for j_diag in self.map_jac_diag(x)])
        # # regularisation objective
        # r = torch.norm(self.weights, p='fro')
        # # return weighted sum of objectives 
        # return self.p_err*e + self.p_drv*s + self.p_reg*r
        pass

# end class LayerDual