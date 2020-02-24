% Octave Oliviers - 16 February 2019
clear all

rng(5) ;

dim_patterns = 1 ;
num_patterns = 2 ;

% create patterns to memorize
patterns = randn(dim_patterns, num_patterns) ;


% build model to memorize patterns
eta    = 1e3 ; % importance of error
gamma  = 1e3 ; % importance of regularization
[W, b] = build_model_shallow(patterns, 'dual', 'tanh', 0, eta, gamma) ;

x = -3:0.1:3 ;

figure()
box on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
plot(x, W'*tanh(x)+b, 'color', [0 0 1], 'linewidth', 1)
plot(patterns, patterns, 'rx')
hold off
title(strcat('eta = ', num2str(eta),', gamma = ', num2str(gamma) ))



%%

% build memory LS-SVM model
%       data            data matrix with instances in columns
%       formulation     primal or dual
%       fun             feature map or kernel function
%       param           parameter of feature map or kernel function

function [W, b] = build_model_shallow(data, formulation, fun, param, eta, gamma)

    [dim_data, num_data] = size(data);
    N = dim_data ;
    P = num_data ;
    
    formulation = lower(formulation);
    
    switch formulation

        case {'primal', 'p'}

            % feature map in each data point
            Phi = tanh(data) ;
            % jacobians of feature map in each of the data points
            F   = jacobian_matrix(data, fun, param) ;
            % dimension of dual space
            D   = size(Phi, 1) ;
            
            % matrices for linear system AX=B
            A = zeros( D+1, D+1 ) ;
            B = zeros( D+1, N ) ;
            
            % LHS
            A( 1:D, 1:D ) = (Phi*Phi') + gamma*(F*F')/eta ;
            A( 1:D, end ) = sum(Phi, 2) ;
            A( end, 1:D ) = sum(Phi, 2) ;
            A( end, end ) = P ;
            
            % RHS
            B( 1:D, : ) = Phi*data' ;
            B( end, : ) = sum(data, 2) ;
            
            % compute hidden parameters L and K
            X = A\B ;
            
            % extract dual variables L and K
            W = X(1:N, :) ;
            b = X(end, :)' ;
            
        case {'dual', 'd'}
                        
            % matrices for linear system AX=B
            A = zeros( P+1, P+1 ) ;   % LHS
            B = zeros( P+1, N ) ;     % RHS
            
            % jacobians of feature map in each of the data points
            F   = jacobian_matrix(data, fun, param) ;
            % kernel matrix
            Phi = tanh(data) ;
            K   = Phi' * pinv(F*F') * Phi ;
            
            % LHS fill first square
            A(1:P, 1:P) = K/gamma + eye(P)/eta ;
            
            % LHS fill last row and column
            A(1:P, end) = 1 ;
            A(end, 1:P) = 1 ;
            
            % RHS fill top
            B(1:P, :) = data' ;
            
            % compute hidden parameters L and K
            X = A\B ;
            
            % extract dual variables L and K
            L = X(1:P, :)' ;
            b = X(end, :)' ;
            
            W = pinv(F*F') * Phi * L' ./gamma ;
    end

    disp("model built")
end