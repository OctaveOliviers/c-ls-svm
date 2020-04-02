% Created  by OctaveOliviers
%          on 2020-03-29 18:59:49
%
% Modified on 2020-03-29 19:35:42

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_patterns    = 2 ;
num_groups      = 6 ;   % number of groups of patterns
num_patterns    = 5 ;   % number of patterns in each group
%
num_test        = 5 ;

% dimension hidden sapce
% dim_hidden        = 4 ;

% parameters of model
num_layers      = 5 ;
[ formulation{1:num_layers} ]   = deal('dual') ;
[ feature_map{1:num_layers} ]   = deal('poly') ;
[ parameter{1:num_layers} ]     = deal([3, 1]) ;
% formulation   = { 'dual',     'dual', 'dual' } ;
% feature_map   = { 'poly',     'poly', 'poly' } ;
% parameter         = { [3, 1],     [3, 1],  [3, 1] } ;

% hyper-parameters
p_err           = 1e4 ; % importance of error
p_reg           = 1e1 ; % importance of regularization
p_drv           = 1e3 ; % importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
% means are spread evenly around origin
z           = exp(i*pi/1)*roots([ 1, zeros(1, num_groups-1), 1]) ;
means       = 5*[ real(z), imag(z) ]' ;
patterns    = means + randn( dim_patterns, num_groups, num_patterns ) ;
% labels        = [1:num_groups] .* ones(10, 1) .* ones(1, 1, num_patterns) ;
[x, y, z]   = ndgrid(-1:2:1) ;
labels      = [ x(1:num_groups) ; y(1:num_groups) ; z(1:num_groups) ] ...
            .* ones(1, 1, num_patterns) ;

patterns    = reshape( patterns, [ dim_patterns, num_groups*num_patterns ] ) ;
labels      = reshape( labels, [ size(labels, 1), num_groups*num_patterns ] ) ;
[ hiddens{1:num_layers-1} ] = deal(labels) ;


% build model
model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg) ;
% train model
model = model.train_explicit( patterns, hiddens ) ;
% visualize model
test = means + randn(dim_patterns, num_groups, num_test ) ;
model.visualize( reshape(test, [dim_patterns, num_groups*num_test] ) ) ;
