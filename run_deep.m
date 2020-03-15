% @Author: OctaveOliviers
% @Date:   2020-03-13 18:40:25
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 18:20:12

clear all
clc

% parameters of patterns
dim_patterns 	= 2 ;
num_patterns 	= 4 ;

% parameters of model
formulation		= {	'primal', 	'primal', 	'primal'} ;
feature_map 	= {	'sign',		'sign', 	'sign'} ;
parameter   	= {	0,			0,			0} ; ;
num_layers  	= 3 ;
% hyper-parameters
p_err  			= 1e4 ;	% importance of error
p_reg  			= 1e1 ;	% importance of regularization
p_drv  			= 1e3 ;	% importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
patterns = 10*rand( dim_patterns, num_patterns ) - 5 ;
% build model
model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg) ;
% train model
model = model.train( patterns ) ;
% visualize model
model.visualize( 10*rand(dim_patterns, 5) - 5 ) ;
