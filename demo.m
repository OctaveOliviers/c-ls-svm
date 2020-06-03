% Created  by OctaveOliviers
%          on 2020-06-03 18:10:23
%
% Modified on 2020-06-03 18:20:56

% Demo for building and training a one-layered C-LS-SVM

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

% initialize random number generator
rng(10) ;

% parameters of memories
dim_memos       = 2 ;
num_memos       = 10 ;
scale_memos     = 17 ; 
shape_memos     = 'S' ;

% (hyper-)parameters of the layer
space           = 'dual' ;          % space to train layer
dim_input       = dim_memories ;    % dimension of the input space
hp_equi         = 1e2 ;             % importance of equilibrium objective
hp_stab         = 1e2 ;             % importance of local stability objective
hp_reg          = 1e-2 ;            % importance of regularization
feat_map        = 'rbf' ;           % chosen feature map or kernel function
feat_map_param  = 8 ;               % parameter of feature map or kernel function




% define the model
model = CLSSVM( ) ;

% add a layer
model = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;

% create memories to store
memories = gen_data_manifold( shape_patterns, scale_patterns, num_patterns, 0 ) ;

% train model
model = model.train( memories ) ;


% select random pattern to start random walk from
% gen_1 = model.generate( memories( :, randi([ 1, num_patterns ]) ), 1000, 0.8) ;
% gen_2 = model.generate( [15 ; -10], 1000, 0.8) ;
% gen_3 = model.generate( [15 ; 10], 1000, 0.8) ;
% gen_4 = model.generate( [-5 ; 15], 1000, 0.8) ;
% gen_5 = model.generate( [-15 ; 0], 1000, 0.8) ;

% walk on manifold
% walk = model.walk_on_manifold( [0; -16], [0; 16], 1 ) ;

% visualize model
model.visualize( [], [] , []  ) ;
% model.visualize( [], [], [ gen_1, gen_2, gen_3, gen_4, gen_5 ] ) ;


% plot singular values of jacobian
% [U, S] = model.jacobian_singular_values() ;
% plot_singular_values( S ) ;