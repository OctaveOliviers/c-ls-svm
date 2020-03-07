% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-07 18:13:07

clear all
clc

rng(6) ;

dim_patterns = 2 ;
num_patterns = 4 ;

% create patterns to memorize
% patterns = [0.5*randn(dim_patterns, num_patterns)+[-2.5; -5], ...
% 			0.5*randn(dim_patterns, num_patterns)+[2.5; -5], ...
% 			0.5*randn(dim_patterns, num_patterns)+[-5;  2.5], ...
% 			0.5*randn(dim_patterns, num_patterns)+[ 5; 2.5], ...
% 			0.5*randn(dim_patterns, num_patterns)+[ 0; 5] ] ;
[X, Y] = meshgrid(-6:3:6, -6:3:6) ;
patterns = [X(:)' ; Y(:)'] ;

% model architecture
formulation = 'dual' ;
feature_map = 'gauss' ;
parameter   = 2 ;
num_layers	= 1 ;

% build model to memorize patterns
p_err  = 1e4 ;	% importance of error
p_reg  = 1e1 ;	% importance of regularization
p_drv  = 1e3 ;	% importance of minimizing derivative


model = Memory_Model_Shallow(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;

model.visualize( 2*randn(dim_patterns, 20) ) ;