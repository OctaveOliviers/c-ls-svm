% Created  by OctaveOliviers
%          on 2020-03-29 17:04:21
%
% Modified on 2020-05-08 08:35:46

clear all
clc

% initialize random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of patterns
dim_patterns    = 2 ;
num_patterns    = 8 ;
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
model = Memory_Model( max_iter, alpha ) ;

% hyper-parameters of layer
p_err = 1e3 ;   % importance of error
p_reg = 1e-1 ;   % importance of regularization
p_drv = 1e2 ;   % importance of minimizing derivative

% model = model.add_layer( { Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) } ) ;
model = model.add_layer( { Layer_Dual( dim_patterns, p_err, p_drv, p_reg, 'rbf', 8) } ) ;

% create deep model
% layer_1 = Layer_Dual  ( 3*dim_patterns, p_err, 1e-2, 1e-2, 'poly', [1, 0] ) ;
% layer_2 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_3 = Layer_Primal( 3*dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_4 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% % layer_3 = Layer_Primal( dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
% model = model.add_layer( { layer_1, layer_2 } ) ;

% create patterns to memorize
switch dim_patterns

    case 1
        % patterns = [ -2, 8] ;
        % patterns = [ -1,-2, 7,8 ] ;
        patterns = [ -1,-2, 7,8, -6] ;

    case 2
        % patterns = [scale_patterns/2; 0] + gen_data_manifold( shape_patterns, scale_patterns, num_patterns, 0.5 ) ;
        patterns = gen_data_manifold( shape_patterns, scale_patterns, num_patterns, 0 ) ;
        manifold = gen_data_manifold( shape_patterns, scale_patterns, 50, 0 ) ;

    otherwise
        error("Cannot simulate more than 2 dimensions, yet.")
end

% train model
model = model.train( patterns ) ;


% select random pattern to start random walk from
% gen_1 = model.generate( patterns( :, randi([ 1, num_patterns ]) ), 1000, 0.8) ;
% gen_2 = model.generate( [15 ; -10], 1000, 0.8) ;
% gen_3 = model.generate( [15 ; 10], 1000, 0.8) ;
% gen_4 = model.generate( [-5 ; 15], 1000, 0.8) ;
% gen_5 = model.generate( [-15 ; 0], 1000, 0.8) ;

% visualize model
model.visualize( [],[] ) ;
% model.visualize( [], [], [ gen_1, gen_2, gen_3, gen_4, gen_5 ] ) ;


% plot singular values of jacobian
% [U, S] = model.jacobian_singular_values() ;
% plot_singular_values( S ) ;

% model.layers{1}

% model.layers{1}.J

% model.layers{1}

% path = model.simulate(0) ;
% E = model.energy_2(path)