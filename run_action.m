% Created  by OctaveOliviers
%          on 2020-03-29 17:04:35
%
% Modified on 2020-06-02 17:45:21

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

% dim_movements = 2 ;
% num_movements = 1 ;
% len_movements = 10 ;

load('data/hello_written.mat') ; 
dim_movements = size(z, 1) ; 
num_movements = 4 ; 
len_movements = size(z, 2) ;

% create patterns to memorize

% initialize random number generator
rng(10)
% movements = 2*randn(dim_movements, num_movements, len_movements) ;
% store 4 movements
movements = zeros(dim_movements, num_movements, len_movements) ;
movements(:, 1, :) = z+[1; -1.5] ; 
movements(:, 2, :) = -z+[-1.5; -1] ; 
movements(:, 3, :) = -z+[6; 3.5] ; 
movements(:, 4, :) = z+[-6; 3] ;


% model architecture
formulation = 'dual' ;
feature_map = 'rbf' ;
parameter   = 2 ;
num_layers  = len_movements-1 ;


% build model to memorize patterns
p_err  = 1e2  ;  % importance of error
p_reg  = 1e-2 ;  % importance of regularization
p_drv  = 1e1  ;  % importance of minimizing derivative


% % build model
% model = Memory_Model_Action(num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg) ;
% % train model
% model = model.train(movements) ;
% % visualize model
% model.visualize( movements(:, :, 1) + randn(dim_movements, num_movements) ) ;


% build model
model = Memory_model( max_iter, alpha ) ;
% add one layer for each step of the movement
for l = 1:num_layers
    model = model.add_layer( formulation, dim_movements, p_err, p_drv, p_reg, feature_map, parameter ) ;
end


% train model


% visualize model
start_simulation = movements( :, :, 1 ) + 0.5 * randn( dim_movements, num_movements, len_movements ) ;

for