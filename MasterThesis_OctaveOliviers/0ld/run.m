% Created  by OctaveOliviers
%          on 2020-03-29 17:04:21
%
% Modified on 2020-06-04 12:58:38

% Experiments on shallow and deep C-LS-SVMs

clear all
clc

% initialize random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of patterns
dim_patterns    = 2 ;
num_patterns    = 10 ;
scale_patterns  = 17 ; 
shape_patterns  = 'S' ;

% model architecture
num_layers      = 1 ;

% model training
max_iter        = 30 ;
alpha           = 0.01 ;

% create model
model = CLSSVM( max_iter, alpha ) ;

% hyper-parameters of layer
p_err_1 = 1e2 ;   % importance of error
p_drv_1 = 1e2 ;   % importance of minimizing derivative
p_reg_1 = 1e-2 ;   % importance of regularization
%
p_err_2 = 1e2 ;   % importance of error
p_drv_2 = 1e2 ;   % importance of minimizing derivative
p_reg_2 = 1e-2 ;   % importance of regularization

% create shallow model
model = model.add_layer( 'dual', dim_patterns, p_err_1, p_drv_1, p_reg_1, 'rbf', 8 ) ;

% create deep model
% layer_1 = Layer_Primal( 3*dim_patterns, p_err_1, p_drv_1, p_reg_1, 'poly', [3, 1] ) ;
% layer_2 = Layer_Primal( dim_patterns, p_err_2, p_drv_2, p_reg_2, 'tanh' ) ;
% model = model.add_layer( { layer_1, layer_2 } ) ;

% create memories to store
switch dim_patterns
    case 1
        memories = [ -6, -1, 7 ] ;

    case 2
        memories = gen_data_manifold( shape_patterns, scale_patterns, num_patterns, 0 ) ;

    otherwise
        error("Cannot simulate more than 2 dimensions, yet.")
end

% train model
model = model.train( memories ) ;

% select random pattern to start random walk from
gen = model.generate( memories( :, randi([ 1, num_patterns ]) ), 1000, 0.8) ;

% walk on manifold
% walk = model.walk_on_manifold( [0; -16], [0; 16], 1 ) ;

% visualize model
model.visualize( [], [] , gen ) ;

% plot singular values of jacobian
% [U, S] = model.jacobian_singular_values() ;
% plot_singular_values( S ) ;