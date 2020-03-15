%% save action in layered LS-SVM formulation - Octave Oliviers 18 February 2020


data = [ 0.4551 0.6509 1.3422 1.9297 1.8606 2.4136 3.1855 3.2661 4.1071 ;
        -0.6175 0.5922 1.1855 0.5835 -0.3151 -0.6722 -0.3885 0.9248 1.5009 ];
    
    
    

    

% build model to memorize patterns
eta    = 1e3 ; % importance of error
gamma  = 1e3 ; % importance of regularization

% for each level
for l = 1:size(data, 2)
    
    [W, b] = build_model_shallow(patterns, 'dual', 'tanh', 0, eta, gamma) ;
    
    
end
    
    



%% test of newton step
clear all
disp(" ") ;

% declare variables
P = 10 ;    % number of patterns
L = 5 ;     % number of layers
I = 20 ;    % number of Newton iterations
g = 1 ;     % regularization parameter
x   = randn(1, P) ;   % patterns to memorize
H   = randn(L, P) ;   % hidden states
Lam = randn(L, P) ;   % Lagrange multipliers
w   = ones(L, 1) ;   % weights vector

cost = @(h_L) sum( (x - h_L).^2, 2 )/2 + g/2 * sum( w.^2, 1 ) ;
phi  = @(x) tanh(x) ;
dphi = @(x) sech(x).^2 ;

% disp("original cost = " + num2str(cost(H(L, :))) ) ;

for i = 1:I
    
    % forward step
    H(1, :) = w(1) * phi(x) ;
    for l = 2:L
        H(l, :) = w(l) * phi( H(l-1, :) ) ;
    end

    % backward step
    Lam(L, :) = x - H(L, :) ;
    for l = L-1:-1:1
        Lam(l, :) = w(l+1) * Lam(l+1, :) .* dphi(H(l, :)) ;
    end

    % weights update
    w = 1/g * sum( L .* phi(H) , 2) ;
    
%     disp("cost in iteration " + num2str(i) + " = " + num2str(cost(H(L, :))) ) ;
    % compute Lagrangian
    lag = cost( H(L, :) ) + sum( Lam(1, :) .* (H(1, :) - w(1)*phi(x)) , 2 ) ;
    for l = 2:L
        lag = lag + sum( Lam(l, :) .* (H(l, :) - w(l)*phi(H(l-1, :))) , 2 ) ;
    end
    
    disp("Lagrangian in iteration " + num2str(i) + " = " + lag ) ;
    
end

disp(" ") ;

%%

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



