% Created  by OctaveOliviers
%          on 2020-03-29 17:04:21
%
% Modified on 2020-04-10 13:22:36

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_in          = 1 ;
dim_out         = 1 ;
num_patterns    = 5 ;
%
num_test        = 5 ;

% aprameters of model
% formulation = 'dual' ; feature_map = 'p' ; parameter = [3, 1] ;
formulation = 'dual' ; feature_map = 'g' ; parameter = 1 ;
% formulation = 'primal' ; feature_map = 'sign' ; parameter = 0 ;
% formulation = 'primal' ; feature_map = 'tanh' ; parameter = 0 ;
num_layers      = 1 ;
% hyper-parameters
p_err           = 1e2 ; % importance of error
p_reg           = 1e-2 ; % importance of regularization
p_drv           = 1e1 ; % importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize

% means are spread evenly around origin
% num_groups    = 30 ;
% z             = exp(i*pi/1)*roots([ 1, zeros(1, num_groups-1), 1]) ;
% patterns      = 5*[ real(z), imag(z) ]' ;
% patterns  = means + randn( dim_patterns, num_groups, num_patterns ) ;
patterns = [1, 2, -0, -4; 1, 2, -2, -1] ;
% patterns = [1, 1, 3, 3, 4, 5, 6, 7] ;

patterns = patterns - mean(patterns) ;

% build model
model = Memory_Model_Encoder_Dual( feature_map, parameter, p_err, p_drv, p_reg ) ;

% train encoder
model = model.train( patterns, dim_out ) ;

% visualize model
model.visualize( ) ;