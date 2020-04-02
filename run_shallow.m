% Created  by OctaveOliviers
%          on 2020-03-29 17:04:21
%
% Modified on 2020-03-29 19:36:02

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_patterns    = 1 ;
num_patterns    = 5 ;
%
num_test        = 5 ;

% aprameters of model
% formulation = 'dual' ; feature_map = 'p' ; parameter = [3, 1] ;
% formulation = 'dual' ; feature_map = 'g' ; parameter = 3 ;
% formulation = 'dual' ; feature_map = 'sign' ; parameter = 0 ;
formulation = 'primal' ; feature_map = 'tanh' ; parameter = 0 ;
num_layers      = 1 ;
% hyper-parameters
p_err           = 1e4 ; % importance of error
p_reg           = 1e1 ; % importance of regularization
p_drv           = 1e3 ; % importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
patterns = 18*rand( dim_patterns, num_patterns ) - 9 ;
% patterns = -10 : 9 : 10 ;
% patterns = [0.5*randn(dim_patterns, num_patterns)+[0; -5], ...
%           0.5*randn(dim_patterns, num_patterns)+[0; +5], ...
%           0.5*randn(dim_patterns, num_patterns)+[-5;  0], ...
%           0.5*randn(dim_patterns, num_patterns)+[ 5; 0] ] ;
% [X, Y] = meshgrid(-6:3:6, -6:3:6) ; patterns = [X(:)' ; Y(:)'] ;

% means are spread evenly around origin
% num_groups    = 6 ;
% z             = exp(i*pi/1)*roots([ 1, zeros(1, num_groups-1), 1]) ;
% means         = 5*[ real(z), imag(z) ]' ;
% patterns  = means + randn( dim_patterns, num_groups, num_patterns ) ;
% labels        = [1:num_groups] .* ones(1, 1, num_patterns) ;

% patterns  = reshape( patterns, [ dim_patterns, num_groups*num_patterns ] ) ;

% build model
model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg ) ;
% train model
model = model.train( patterns ) ;
% visualize model
% model.visualize( means + 3*randn ) ;
% test = means + randn(dim_patterns, num_groups, num_test ) ;
% model.visualize( reshape(test, [dim_patterns, num_groups*num_test] ) ) ;
model.visualize() ;

% isequal(model.W, model.W')

% check 
% [E, err, eigv] = model.energy( patterns ) ;
% err
% max(eigv, [], 'all')