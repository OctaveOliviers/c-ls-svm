% Created  by OctaveOliviers
%          on 2020-05-11 17:32:47
%
% Modified on 2020-06-04 13:00:00

% Experiment to train a deep C-LS-SVM by explicitly assigning the hidden states

clear all
clc

% initialize random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of patterns
dim_patterns    = 2 ;
dim_hidden      = 3 ;
num_patterns    = 20 ;
num_groups      = 4 ;
scale_patterns  = 15 ; 
shape_patterns  = 'g' ;

% model architecture
num_layers      = 1 ;

% create model
model = CLSSVM( ) ;

% hyper-parameters of layer
p_err = 1e4 ;   % importance of error
p_reg = 1e1 ;   % importance of regularization
p_drv = 1e3 ;   % importance of minimizing derivative

% model = model.add_layer( 'primal', dim_patterns, p_err, p_drv, p_reg, 'sigmoid' ) ;
% model = model.add_layer( { Layer_Dual( dim_patterns, p_err, p_drv, p_reg, 'poly', [4, 1] ) } ) ;

% create deep model
for l = 2:num_layers
    % model = model.add_layer( 'primal', dim_hidden, p_err, p_drv, p_reg, 'tanh' ) ;
    model = model.add_layer( 'dual', dim_patterns, p_err, p_drv, p_reg, 'poly', [1, 1] ) ;
end
% model = model.add_layer( 'primal', dim_patterns, p_err, p_drv, p_reg, 'tanh' ) ;
model = model.add_layer( 'dual', dim_patterns, p_err, p_drv, p_reg, 'poly', [3, 1] ) ;


% create memories to store
z           = exp(i*pi/4)*roots([ 1, zeros(1, num_groups-1), 1]) ;
means       = scale_patterns*[ real(z), imag(z) ]' ;
memories    = means + 2*randn( dim_patterns, num_groups, num_patterns ) ;
% choose hidden representations as vertices of 3D unit cube
hiddens     = cell(1, num_layers-1) ;
[x, y, z]   = ndgrid(-1:2:1) ;
labels      = [ x(1:num_groups) ; y(1:num_groups)].* ones(1, 1, num_patterns) ;

memories    = reshape( memories, [ dim_patterns, num_groups*num_patterns ] ) ;
labels      = reshape( labels, [ size(labels, 1), num_groups*num_patterns ] ) ;
hiddens(1:num_layers-1) = { labels } ;

% train model
model = model.train( memories, hiddens ) ;


model.visualize( ) ;