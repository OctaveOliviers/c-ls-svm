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

max_iter = 100 ;
alpha = 1 ;

% create model
model = CLSSVM( max_iter, alpha ) ;

% hyper-parameters of layer
p_err = 1e2 ;   % importance of error
p_reg = 1e-2 ;   % importance of regularization
p_drv = 1e1 ;   % importance of minimizing derivative

% model = model.add_layer( 'primal', 1, p_err, p_drv, p_reg, 'sigmoid' ) ;
model = model.add_layer( { Layer_Dual( 1, p_err, p_drv, p_reg, 'rbf', 2 ) } ) ;

% model = model.add_layer( { Layer_Dual( 1, p_err, p_drv, p_reg, 'poly', [3,1] ) } ) ;
% model = model.add_layer( 'primal', 1, p_err, p_drv, p_reg, 'tanh' ) ;

% create patterns to memorize
patterns = 0.5*[-15 -8, -7, -1,-2, 7, 15, 16 ] ;
        
% train model
model = model.train( patterns ) ;

% visualize update equation
% model.visualize( ) ;
x = -10:0.1:10 ;
f1 = model.simulate_one_step(x) ;
% visualize convergence map
f = model.simulate(x) ;
conv = [] ;
for i = 1:length(f)
    conv(end+1) = f{i}(end) ;
end
plot_1d_map(patterns, x, f1)
plot_1d_map(patterns, x, conv)


% f_plot = readtable('../data/cae_1d.csv');
% plot_1d_map(patterns, x, [f_plot.Var3(2); f_plot.Var3(2:end)])
% plot_1d_map(patterns, x, [f_plot.Var2(2); f_plot.Var2(2:end)])

