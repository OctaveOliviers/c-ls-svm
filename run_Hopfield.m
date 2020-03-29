% Created  by OctaveOliviers
%          on 2020-03-29 17:04:27
%
% Modified on 2020-03-29 19:35:47

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