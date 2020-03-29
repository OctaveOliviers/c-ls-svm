% @Author: OctaveOliviers
% @Date:   2020-03-20 08:25:48
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-20 09:09:26

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

warning( 'Results are not correct.' )

% parameters of patterns
dim_patterns    = 1 ;
num_patterns    = 2 ;
%
num_test        = 5 ;

% initialize random number generator
rng(10) ;

% create patterns to memorize
patterns = randi( [0, 1] , dim_patterns, num_patterns ) ;

% build model
model = Hopfield_Network( 'sign' ) ;
% train model
model = model.train( patterns ) ;
% visualize model
model.visualize() ;