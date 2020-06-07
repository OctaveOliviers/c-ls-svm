% Created  by OctaveOliviers
%          on 2020-03-29 17:04:21
%
% Modified on 2020-06-04 13:10:52

clear all
clc

% initialize random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of patterns
dim_patterns    = 2 ;
num_patterns    = 250 ;
scale_patterns  = 17 ; 
shape_patterns  = 's' ;

% parameters of visualization
num_test        = 5 ;

% model architecture
num_layers      = 1 ;

% model training
max_iter        = 15 ;
alpha           = 1 ;

% create model
model = CLSSVM( max_iter, alpha ) ;

% hyper-parameters of layer
p_err = 1e2 ;   % importance of error
p_reg = 1e-2 ;   % importance of regularization
p_drv = 1e2 ;   % importance of minimizing derivative

% model = model.add_layer( 'primal', dim_patterns, p_err, p_drv, p_reg, 'sigmoid' ) ;
model = model.add_layer( { Layer_Dual( dim_patterns, p_err, p_drv, p_reg, 'rbf', 8 ) } ) ;

% create deep model
% layer_1 = Layer_Dual  ( dim_patterns, 1e3, 1e-2, 1e1, 'poly', [5, 1]) ;
% layer_2 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_2 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_3 = Layer_Primal( 3*dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_4 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_3 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% model = model.add_layer( { layer_1, layer_2 } ) ;

% create patterns to memorize
switch dim_patterns

    case 1
        patterns = [ -8, -7, -1,-2, 7,8 ] ;
        
    case 2
        patterns = gen_data_manifold( shape_patterns, scale_patterns, num_patterns, 0.5 ) ;
        
    otherwise
        error("Cannot simulate more than 2 dimensions, yet.")
end

% train model
model = model.train( patterns ) ;

% walk on manifold
walk = model.walk_on_manifold( [0; -16], [0; 16], 1 ) ;

% visualize model
model.visualize( [], [] , [] , walk ) ;
% model.visualize( [], [], [ gen_1, gen_2, gen_3, gen_4, gen_5 ] ) ;


% plot singular values of jacobian
% [U, S] = model.jacobian_singular_values() ;
% plot_singular_values( S ) ;

% model.layers{1}

% model.layers{1}.J

% model.layers{1}

% path = model.simulate(0) ;
% E = model.energy_2(path)