% Created  by OctaveOliviers
%          on 2020-06-03 18:10:23
%
% Modified on 2020-06-03 19:11:25

% Demo for building and training a one-layered C-LS-SVM

clear all
clc

% add the folders to the Matlab path
addpath( './models/' )
addpath( './util/' )

% initialize random number generator
rng(10) ;

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