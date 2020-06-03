# Master Thesis
## Deep Contractive Least Squares Support Vector Machines for associative memory

This code is free to use and modify. 

### Background

A C-LS-SVM is a dynamical system to model auto-associative memory.
Thus, it is a dynamical system that stores memories as stable equilibria.


### Software

#### Requirements

- Matlab (https://www.mathworks.com/products/matlab.html).
- add the folders to the Matlab path

    model = CLSSVM( ) ;
    

#### Build a C-LS-SVM

A C-LS-SVM model consists of several layers. Each layer has either an explicit or an implicit feature map.

##### Define the model

The easiest way to define a C-LS-SVM is
```
model = CLSSVM( ) ;
```
The model 

##### Add layers

To add a layer, you need to define 7 parameters:
1. the space in which to train the layer `<space>` (string);
1. the size of the input space for that layer `<dim_input>` (integer);
1. the hyper-parameter for the equilibrium objective `<hp_equi>` (float);
1. the hyper-parameter for the local stability objective objective `<hp_stab>` (float);
1. the hyper-parameter for the regularization `<hp_reg>` (float);
1. the chosen feature map or kernel function `<feat_map>` (string);
1. the parameters of the feature map or kernel function `<feat_map_param>` (float/integer).

```
model = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;
```
You can keep adding as many layers as you need by simply repeating the previous command.

#### Train a C-LS-SVM 

##### Create the memories to store

The memories are defined by two parameters
1. the dimension of the input space `<dim_memos>` (integer)
1. The number of memories to store `<num_memos>` (integer)
The sofware 

You can either use you own memories, e.g.
```
memories = randn(dim_memos, num_memos) ;
```
or generate
1. the space in which to train the layer `<space>` (string)
1. the size of the input space for that layer `<dim_input>` (integer)
1. the hyper-parameter for the equilibrium objective `<hp_equi>` (float)
1. the hyper-parameter for the local stability objective objective `<hp_stab>` (float)
1. the hyper-parameter for the regularization `<hp_reg>` (float)
1. the chosen feature map or kernel function `<feat_map>` (string)
1. the parameters of the feature map or kernel function `<feat_map_param>` (float/integer)
```
memories = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;
```

##### Explicitly assigning the hidden states

##### Implicitly learning the hidden states with the parameters

#### Visualize a C-LS-SVM

#### Demo