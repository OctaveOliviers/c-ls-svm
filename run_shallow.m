% Created  by OctaveOliviers
%          on 2020-04-11 14:54:28
%
% Modified on 2020-04-16 16:29:26

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of patterns
dim_patterns    = 1 ;
num_patterns    = 50 ;
scale_patterns  = 18 ; 

% parameters of visualization
num_test        = 5 ;

% model architecture
num_layers      = 1 ;

% model training
max_iter   = 3 ;
alpha      = 1 ;

% create model
model = Memory_Model( max_iter, alpha ) ;

% parameters of layer
% space = 'dual' ; phi = 'p' ; theta = [3, 1] ;
% space = 'dual' ; phi = 'rbf' ; theta = 3 ;
% space = 'primal' ; phi = 'sign' ; theta = 0 ;
space = 'primal' ; phi = 'sign' ; theta = 0 ;

% hyper-parameters of layer
p_err = 1e3 ;   % importance of error
p_reg = 1e-1 ;   % importance of regularization
p_drv = 1e1 ;   % importance of minimizing derivative

layer_1 = Layer_Dual  ( dim_patterns, p_err, p_drv, p_reg, 'poly', [1, 0] ) ;
layer_2 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;

% model = model.add_layer( { Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) } ) ;
% model = model.add_layer( { Layer_Dual( dim_patterns, p_err, p_drv, p_reg, 'rbf', 2 ) } ) ;
%
model = model.add_layer( { layer_1, layer_2 } ) ;


% initialize random number generator
rng(10) ;

% create patterns to memorize
switch dim_patterns

    case 1
        % patterns = [ -4 , -2 , 0.5 , 1 , 3 ] ;
        patterns = [ 1 , 4 ] ;

    case 2
        patterns = gen_data_manifold( 'S', scale_patterns, num_patterns, 1 ) ;

    otherwise
        error("Cannot simulate more than 2 dimensions, yet.")
end

% train model
model = model.train( patterns ) ;

% visualize model
model.visualize(  ) ;

model.layers{1}

% model.layers{1}

% path = model.simulate(0) ;
% E = model.energy_2(path)