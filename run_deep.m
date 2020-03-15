% @Author: OctaveOliviers
% @Date:   2020-03-13 18:40:25
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-14 18:59:26

clear all
clc

% parameters of patterns
dim_patterns 	= 2 ;
num_patterns 	= 4 ;

% parameters of model
formulation		= {	'primal', 	'pimral', 	'primal'} ;
feature_map 	= {	'tanh',		'tanh', 	'tanh'} ;
parameter   	= 0 ;
num_layers  	= 3 ;
% hyper-parameters
p_err  			= 1e4 ;	% importance of error
p_reg  			= 1e1 ;	% importance of regularization
p_drv  			= 1e3 ;	% importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
patterns = 3*randn( dim_patterns, num_patterns ) ;
% build model
model = Memory_Model_Deep(	num_layers, 
							formulation, 
							feature_map, 
							parameter, 
							p_err, p_drv, p_reg) ;
% train model
model = model.train( patterns ) ;
% visualize model
model.visualize( randn( dim_patterns, 5 ) ) ;