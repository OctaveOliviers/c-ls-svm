% Octave Oliviers - 20 December 2019
clear all

% rng(5) ;

dim_patterns = 1 ;
num_patterns = 2 ;


% create patterns to memorize
% patterns = [0.1*randn(dim_patterns, num_patterns)-3 0.1*randn(dim_patterns, num_patterns)+3];
% patterns = randn(dim_patterns, num_patterns) ;
patterns = [-0.3430    1.6663] ;

% build model to memorize patterns
eta    = 1e0 ;
lambda = 1e5 ;
gamma  = 1e3 ;
% num_layers = 2 ; fun    = { 'poly', 'tanh'} ; param  = {1, 0} ;
num_layers = 1 ; fun    = {'tanh'} ; param  = {3} ;
[W, b] = build_model_b(patterns, num_layers, 'primal', fun, param, lambda, eta, gamma);

x = -5:0.1:5 ;
f = x ;
for l = 1:num_layers
    [~, phi] = kernel_matrix(f, fun{l}, param{l}) ;
    %f = W(l, :, :)'*phi + b(l, :, :) ; % should be this line
    f = W{l}'*phi + b{l} ; 
end


figure('position', [100 100 550 500])
box on
grid on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
plot(x, f, 'color', [0 0 1], 'linewidth', 1)
plot(patterns, patterns, 'rx', 'linewidth', 2)
xticks(min(x):max(x))
yticks(min(x):max(x))
axis equal
hold off
title(strcat('eta = ', num2str(eta),', lambda = ', num2str(lambda), ', gamma = ', num2str(gamma) ))



%% small test
clear all
rng(6)

size = 8 ;
prec = size/50 ;
x = -size:prec:size ;
y = -size:prec:size ;
[X, Y] = meshgrid(x, y) ;
X_vec = X(:)' ;
Y_vec = Y(:)' ;

num_dim = 2 ;
W1 = randn(num_dim, num_dim) ;
b1 = randn(num_dim, 1) ;
W2 = 3*randn(num_dim, num_dim) ;
b2 = randn(num_dim, 1) ;

num_steps = 100 ;
x0 = randn(2, 1) ;
% x0 = [-1; -2] ;
path = zeros(num_dim, num_steps) ;
for n = 1:num_steps
    
    path(:, n) = x0 ;
    
    x0 = W2' * tanh( W1'*(x0).^3 + b1 ) + b2 ;
    
end

F = W2' * tanh( W1'*([X_vec; Y_vec].^3) + b1 ) + b2 ;
f1 = reshape(F(1, :), [length(x), length(y)]) ;
f2 = reshape(F(2, :), [length(x), length(y)]) ;

% lines of intersection
c1 = contour(x,y,X-f1,[0, 0], 'linewidth', 3) ;
c2 = contour(x,y,Y-f2,[0, 0], 'linewidth', 3) ;


figure
hold on
s1 = surf(x, y, X) ;
s1.FaceAlpha = 0.2 ;
s1.EdgeAlpha = 0.3 ;
s2 = surf(x, y, Y) ;
s2.FaceAlpha = 0.2 ;
s2.EdgeAlpha = 0.3 ;
fx = surf(x, y, f1) ;
fx.FaceAlpha = 0.3 ;
fx.EdgeAlpha = 0.4 ;
fy = surf(x, y, f2) ;
fy.FaceAlpha = 0.3 ;
fy.EdgeAlpha = 0.4 ;
plot3(c1(1, :), c1(2, :), c1(1, :), 'linewidth', 2)
plot3(c2(1, :), c2(2, :), c2(2, :), 'linewidth', 2)
plot3(path(1, 1:end-1), path(2, 1:end-1), path(1, 2:end), 'linewidth', 2)
plot3(path(1, 1:end-1), path(2, 1:end-1), path(2, 2:end), 'linewidth', 2)
plot3(x0(1, :), x0(2, :), x0(1, :), 'x', 'markersize', 10, 'linewidth', 3)
plot3(x0(1, :), x0(2, :), x0(2, :), 'x', 'markersize', 10, 'linewidth', 3)
hold off
axis equal
xlim([-size, size])
ylim([-size, size])
zlim([-size, size])
view([30, 30, 20])


%%

% build memory LS-SVM model
%       data            data matrix with instances in columns
%       formulation     primal or dual
%       fun             feature map or kernel function
%       param           parameter of feature map or kernel function

function [W, b] = build_model_b(data, num_layers, formulation, fun, param, lambda, eta, gamma)

    assert( length(fun)   == num_layers ) ;
    assert( length(param) == num_layers ) ;

    [dim_data, num_data] = size(data);
    N = dim_data ;
    P = num_data ;
    L = num_layers ;
    
    formulation = lower(formulation);
    
    switch formulation

        case {'primal', 'p'}
            
            % compute dimension of dual space
            [~, phi] = kernel_matrix(data, fun{1}, param{1}) ;
            D = size(phi, 1) ;

            W = cell(L, 1) ;
            b = cell(L, 1) ;
            
            Z = data ;
            for l = 1:L
                
                % matrices for linear system AX=B
                A = zeros( D+1, D+1 ) ;   % LHS
                B = zeros( D+1, N ) ;     % RHS

                % jacobians of feature map in each of the data points
                F   = jacobian_matrix(Z, fun{l}, param{l}) 
                % kernel matrix
                [~, phi] = kernel_matrix(Z, fun{l}, param{l}) ;

                % LHS fill first square
                A(1:D, 1:D) = (phi*phi') + gamma*(F*F')/lambda + eta*eye(D)/lambda ;

                % LHS fill last row and column
                A(1:D, end) = sum(phi, 2) ;
                A(end, 1:D) = sum(phi, 2)' ;

                % LHS fill last element
                A(end, end) = P ;
                
                % RHS fill 
                if (l < L)
                    T = randn(size(Z)) ;
                    B(1:D, :) = phi*T' ; 
                    B(end, :) = sum(T, 2)' ;
                else
                    B(1:D, :) = phi*data' ;
                    B(end, :) = sum(data, 2)' ;
                end
                
                % compute parameters W and b
                X = A\B ;

                W{l} = X(1:D, :);
                b{l} = X(end, :)' ;
                
                Z = W{l}'*phi + b{l} ; % should be this line
            end
            
            
        case {'dual', 'd'}
            
            % compute dimension of dual space
            [~, phi] = kernel_matrix(data, fun{1}, param{1}) ;
            D = size(phi, 1) ;
            
            W = cell(L, 1) ;
            b = cell(L, 1) ;
            
            Z = data ;
            for l = 1:L
                
                % matrices for linear system AX=B
                A = zeros( (N+1)*P+1, (N+1)*P+1 ) ;   % LHS
                B = zeros( (N+1)*P+1, N ) ;           % RHS

                % jacobians of feature map in each of the data points
                J   = jacobian_matrix(Z, fun{l}, param{l}) ;
                % kernel matrix
                [K, phi] = kernel_matrix(Z, fun{l}, param{l}) ;

                % LHS fill first square
                A(1:P, 1:P) = K/eta + eye(P)/lambda ;

                % LHS fill first row and column
                A(1:P, P+1:end-1) = phi'*J./eta ;
                A(P+1:end-1, 1:P) = J'*phi./eta ;

                % LHS fill last column of first row
                A(1:P, end) = ones(P, 1) ;
                % fill last row of first column
                A(end, 1:P) = ones(1, P) ;

                % LHS fill large square lower right
                A(P+1:end-1, P+1:end-1) = J'*J./eta + eye(N*P, N*P)./gamma ;     

                % RHS fill top
                if (l < L), B(1:P, :) = randn(size(Z))' ;
                else, B(1:P, :) = data' ; end
                    
                % compute hidden parameters H and K
                X = A\B ;

                % extract dual variables H and K
                H = X(1:P, :)' ;
                K = X(P+1:end-1, :)' ;
                
                b{l} = X(end, :)' ;
                W{l} = (phi*H' + J*K')./eta ;
               
                Z = W{l}'*phi + b{l} ; % should be this line
            end
                        
    end

    disp("model built")
end