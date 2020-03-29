% @Author: OctaveOliviers
% @Date:   2020-03-28 12:26:03
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-29 09:39:53

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_patterns 	= 1 ;
num_patterns 	= 4 ;


% hyper-parameters
p_err  		= 1e4 ;	% importance of error
p_reg  		= 1e1 ;	% importance of regularization
p_drv  		= 1e3 ;	% importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
X = [ 1, 3 ] ;

% build models
level_1 	= build_model( 1, 'dual', 'poly', [1, 1], p_err, p_drv, p_reg ) ;
level_2 	= build_model( 1, 'primal', 'tanh', 0, p_err, p_drv, p_reg ) ;

% train models
level_1 	= level_1.train( X ) ;
level_2 	= level_2.train( X ) ;

% visualize model
model 		= Memory_Model_Deep( {level_1, level_2} ) ;
model.visualize() ;

% % compute error
% E = model1.error( X ) ;

% % train model
% model2 = model1.train( X-E, X ) ;
% % visualize model
% model2.visualize() ;