% Created  by OctaveOliviers
%          on 2020-03-29 17:04:25
%
% Modified on 2020-04-11 14:51:49

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_X   = 1 ;
num_X   = 4 ;
% parameters for training deep model
max_it  = 10 ;
alpha   = 1 ;

% hyper-parameters
p_err   = 1e2 ; % importance of error
p_reg   = 1e-2 ; % importance of regularization
p_drv   = 1e1 ; % importance of minimizing derivative

num_lay = 2;
formul  = { "d", "p" } ;
phis    = { "poly", "tanh" } ;
thetas  = { [1, 0], 0 } ;

% initialize random number generator
rng(10) ;

% create patterns to memorize
X = [ 1, 4 ] ;

% visualize model
% model = Memory_Model_Deep( {level_1, level_2} ) ;
%
model = build_model( num_lay, formul, phis, thetas, p_err, p_drv, p_reg, alpha, max_it ) ;

model = model.train_implicit( X ) ;
model.visualize() ;

% % compute error
% E = model1.error( X ) ;

% % train model
% model2 = model1.train( X-E, X ) ;
% % visualize model
% model2.visualize() ;


% % visualize Lagrangian as a fuction of H
% [X, Y] = meshgrid (-5:1:5) ;
% X_vec = X(:) ;
% Y_vec = Y(:) ;
% L = zeros(1, size(X(:), 1)) ;
% for i = size(X_vec, 1)
%     for j = size(Y_vec, 1)
%         L() ;
%     end
% end
