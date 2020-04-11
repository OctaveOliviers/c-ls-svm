% Created  by OctaveOliviers
%          on 2020-04-11 14:54:28
%
% Modified on 2020-04-11 22:09:25

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_patterns    = 1 ;
num_patterns    = 4 ;
scale_patterns  = 18 ; 

% parameters of visualization
num_test        = 5 ;

% model architecture
num_layers      = 1 ;

% model training
max_iter   = 10 ;
alpha      = 0.001 ;

% create model
model = Memory_Model( max_iter, alpha ) ;

% parameters of layer
% space = 'dual' ; phi = 'p' ; theta = [3, 1] ;
% space = 'dual' ; phi = 'rbf' ; theta = 3 ;
% space = 'primal' ; phi = 'sign' ; theta = 0 ;
space = 'primal' ; phi = 'tanh' ; theta = 0 ;
% hyper-parameters of layer
p_err = 1e2 ;   % importance of error
p_reg = 1e-2 ;   % importance of regularization
p_drv = 1e1 ;   % importance of minimizing derivative

layer_1 = Layer_Dual  ( 'poly', [1, 0], p_err, p_drv, p_reg ) ;
layer_2 = Layer_Primal( 'tanh', 0,      p_err, p_drv, p_reg ) ;

model = model.add_layer( { layer_1, layer_2 } ) ;


% initialize random number generator
rng(10) ;

% create patterns to memorize
% patterns = scale_patterns*rand( dim_patterns, num_patterns ) - scale_patterns/2 ;
patterns = [1, 4] ;

% means are spread evenly around origin
% num_groups    = 6 ;
% z             = exp(i*pi/1)*roots([ 1, zeros(1, num_groups-1), 1]) ;
% means         = 5*[ real(z), imag(z) ]' ;
% patterns  = means + randn( dim_patterns, num_groups, num_patterns ) ;
% labels        = [1:num_groups] .* ones(1, 1, num_patterns) ;

% patterns  = reshape( patterns, [ dim_patterns, num_groups*num_patterns ] ) ;

% build model
% model = build_model( num_layers, space, phi, theta, p_err, p_drv, p_reg ) ;
% train model
model = model.train( patterns ) ;


% visualize model


% model.visualize( means + 3*randn ) ;
% test = means + randn(dim_patterns, num_groups, num_test ) ;
% model.visualize( reshape(test, [dim_patterns, num_groups*num_test] ) ) ;
model.visualize( scale_patterns*rand( dim_patterns, num_test ) - scale_patterns/2  ) ;