% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-12 10:10:05

clear all
clc

rng(6) ;

dim_patterns = 2 ;
num_patterns = 8 ;

% create patterns to memorize
% patterns = 3*randn(dim_patterns, num_patterns) ;
patterns = [0.5*randn(dim_patterns, num_patterns)+[-2.5; -5], ...
			0.5*randn(dim_patterns, num_patterns)+[2.5; -5], ...
			0.5*randn(dim_patterns, num_patterns)+[-5;  2], ...
			0.5*randn(dim_patterns, num_patterns)+[ 0; 5] ] ;
% [X, Y] = meshgrid(-6:3:6, -6:3:6) ;
% patterns = [X(:)' ; Y(:)'] ;

% model architecture
% formulation = 'dual' ; feature_map = 'p' ; parameter = [5, 10] ;
% formulation = 'dual' ; feature_map = 'g' ; parameter = 20 ;
% formulation = 'primal' ; feature_map = 'sign' ; parameter = 0 ;
formulation = 'primal' ; feature_map = 'tanh' ; parameter = 0 ;
num_layers	= 1 ;

% build model to memorize patterns
p_err  = 1e4 ;	% importance of error
p_reg  = 1e1 ;	% importance of regularization
p_drv  = 1e3 ;	% importance of minimizing derivative


model = Memory_Model_Shallow(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;

model.visualize( 2*randn(dim_patterns, 10) ) ;