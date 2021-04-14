clear all
clc

% add the folders to the Matlab path
addpath( './models/' )
addpath( './util/' )

x = [2, 7, -7, 1] ;

% (hyper-)parameters of the layer
space           = 'primal' ;
dim_input       = size(x,1) ;
p_equi          = 1e2 ;
p_stab          = 1e2 ;
p_reg           = 1e-2 ;
feat_map        = 'tanh' ;
feat_map_param  = [3,1] ;

% define the model
model = CLSSVM( ) ;

% add a layer
model = model.add_layer( space, dim_input, p_equi, p_stab, p_reg, feat_map, feat_map_param ) ;

% train model
model = model.train( x ) ;

% visualize trained model
model.visualize( ) ;