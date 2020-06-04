## Master Thesis
# Deep Contractive Least Squares Support Vector Machines for associative memory

This code is free to use and modify. 


## Background

LS-SVM = Least Squares Support Vector Machine

C-AE = contractive autoencoder

C-LS-SVM = Contractive Least Squares Support Vector Machine

A C-LS-SVM is a dynamical system to model auto-associative memory.
Specifically, it is a dynamical system that stores memories as locally stable equilibria.
This way, as the state of the dynamical system evolves, it progressively converges to one of the stored memories.
A C-LS-SVM integrates the typical contraction of a C-AE into the LS-SVM framework.

## Software

### Requirements
- Matlab (https://www.mathworks.com/products/matlab.html).
- Add the folders to the Matlab path as
```
    addpath( './models/' )
    addpath( './util/' )
    addpath( './data/' )
```

### Build a C-LS-SVM
A C-LS-SVM model consists of several layers. Each layer has either an explicit or an implicit feature map.

#### Define the model
The easiest way to define a C-LS-SVM is
```
    model = CLSSVM( ) ;
```
The object `model` then contains all the necessary functions to train and simulate the C-LS-SVM.

#### Add layers
A layer is defined by seven parameters:
1. the space in which to train the layer `space` (string) ;
1. the size of the input space for that layer `dim_input` (integer) ;
1. the hyper-parameter for the equilibrium objective `hp_equi` (float) ;
1. the hyper-parameter for the local stability objective objective `hp_stab` (float) ;
1. the hyper-parameter for the regularization `hp_reg` (float) ;
1. the chosen feature map or kernel function `feat_map` (string) ;
1. the parameters of the feature map or kernel function `feat_map_param` (float/integer).

```
    model = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;
```
You can keep adding as many layers as you need by repeating the previous command.


### Train a C-LS-SVM 

##### Create the memories to store
The memories are defined by two parameters:
1. the dimension of the input space `dim_memos` (integer) ;
1. the number of memories to store `num_memos` (integer).

The sofware assumes that the matrix of memories is of size (`dim_memos`, `num_memos`)

You can either use you own memories, e.g.
```
    memories = randn( dim_memos, num_memos ) ;
```
or generate the memories along a specific manifold with three parameters:
1. the scale of the largest memory `scale_memos` (float) ;
1. the shape of the manifold `shape_memos` (string) ;
1. the std of the Gaussian noise on each memory `noise_memos` (float).

```
    memories = gen_data_manifold( shape_memos, scale_memos, num_memos, noise_memos ) ;
```

#### Shallow C-LS-SVM
If the C-LS-SVM only contains one layer, train it as follows
```
    model = model.train( memories ) ;
```

#### Deep C-LS-SVM
If the C-LS-SVM contains `L` layers, there are two alternatives to train the model.

##### 1. Explicitly assigning the hidden states
Define a cell `H` that contains the `L-1` hidden states, and train the model as follows
```
    model = model.train_explicit( memories, H ) ;
```
The model automatically assigns the states in the first and last layers to the memories to store.

##### 2. Implicitly learning the hidden states with the parameters
Train a deep C-LS-SVM that learns good hidden states by itself as follows
```
    model = model.train( memories ) ;
```
The software applies gradient descent to optimize the deep Lagrange function over the hidden states.

### Generative C-LS-SVM 
The generation process consists of a random walk on the data manifold. Therefore, define three parameters:
1. the position to start the walk `start_pos` (vector) ;
1. the number of samples to generate `num_gen` (integer) ;
1. the step size of the random walk `step_size` (float).

```
    gen_memos = model.generate( start_pos, num_gen, step_size ) ;
```

It is also possible to walk on the manifold from one point to another. Therefore, define three parameters:
1. the position to start the walk `start_walk` (vector) ;
1. the position to end the walk `end_walk` (vector) ;
1. the step size of the walk `step_size` (float).
```
    walk = model.walk_on_manifold( start_walk, end_walk, step_size ) ;
```

### Visualize a C-LS-SVM
It is possible to visualize the trained C-LS-SVM as long as `dim_memos <= 2`. Simply run
```
    model = model.visualize( ) ;
```
which will show the memories in the state space, and the dynamics that the C-LS-SVM learned.

To also see the generated data samples
```
    model = model.visualize( [], [], gen_memos  ) ;
```
or the walk on the manifold
```
    model = model.visualize( [], [], [], walk]  ) ;
```

### Demo
We have gathered the most important commands in the demo.m file.
```
    % Demo for building and training a one-layered C-LS-SVM

    % parameters of memories
    dim_memos       = 2 ;
    num_memos       = 100 ;
    scale_memos     = 17 ; 
    shape_memos     = 'S' ;
    noise_memos     = 0.5 ;

    % (hyper-)parameters of the layer
    space           = 'dual' ;          % space to train layer
    dim_input       = dim_memos ;       % dimension of the input space
    hp_equi         = 1e2 ;             % importance of equilibrium objective
    hp_stab         = 1e1 ;             % importance of local stability objective
    hp_reg          = 1e-2 ;            % importance of regularization
    feat_map        = 'rbf' ;           % chosen feature map or kernel function
    feat_map_param  = 4 ;               % parameter of feature map or kernel function


    % define the model
    model = CLSSVM( ) ;

    % add a layer
    model = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;

    % create memories to store
    memories = gen_data_manifold( shape_memos, scale_memos, num_memos, noise_memos ) ;

    % train model
    model = model.train( memories ) ;

    % visualize trained model
    model.visualize( ) ;

    % generate 100 new data points, starting from a random memory (step size of random walk)
    num_gen     = 1000 ;
    step_size   = 1 ;
    gen_memos   = model.generate( memories( :, randi([ 1, num_memos ]) ), num_gen, step_size) ;

    % visualize generated data points
    model.visualize( [], [], gen_memos ) ;

    % walk on manifold
    start_walk = [0; -16] ;
    end_walk   = [0; 16] ;
    step_size  = 1
    walk = model.walk_on_manifold( start_walk, end_walk, step_size ) ;

    % visualize the walk on the manifold
    model.visualize( [], [] , [] , walk ) ;
```

This small piece of code generates three images, namely
| **Trained model** |  **Generated samples** |  **Walk on the manifold** |
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/OctaveOliviers/master-thesis/blob/master/figs/demo-out-1.jpg)  |  ![](https://github.com/OctaveOliviers/master-thesis/blob/master/figs/demo-out-2.jpg) |  ![](https://github.com/OctaveOliviers/master-thesis/blob/master/figs/demo-out-3.jpg)