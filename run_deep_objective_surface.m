% Created  by OctaveOliviers
%          on 2020-03-29 17:04:25
%
% Modified on 2020-04-11 14:51:50

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_X   = 1 ;
num_X   = 4 ;
% parameters for training deep model
max_it  = 10 ;
alpha   = 1 ;

% hyper-parameters
p_err   = 1e2 ; % importance of error
p_reg   = 1e-2 ; % importance of regularization
p_drv   = 1e1 ; % importance of minimizing derivative

num_lay = 2;
formul  = { "d", "p" } ;
phis    = { "poly", "tanh" } ;
thetas  = { [1, 0], 0 } ;

% initialize random number generator
rng(10) ;

% create patterns to memorize
X = [ 1, 4 ] ;

% visualize model
model = build_model( num_lay, formul, phis, thetas, p_err, p_drv, p_reg, alpha, max_it ) ;

[model, param_path] = model.train_implicit( X ) ;
model.visualize() ;