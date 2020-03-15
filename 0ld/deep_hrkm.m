% @Author: OctaveOliviers
% @Date:   2020-02-24 18:47:50
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-14 18:39:39
%
% deep dynamical system for memorizing patterns 
% Octave Oliviers - 24nd February 2020

clear all

% DECLARE PARAMETERS
% for the data
N   = 1 ; 		% number of neurons = dimension of the data
P   = 2 ;		% number of patterns
D   = N ;		% dimension of feature space
% for the network
L   = 1 ;		% number of layers
eta = 1e3 ;		% importance of regularization
bet = 1e2 ;		% importance of flat derivative
gam = 1e2 ;		% importance of zero error
phi = {'tanh'} ;
param = {0} ;


% BUILD DATASET
% X = zeros(N, P) ;
X = [1 2] ;


% INITIALIZE NETWORK
H   = randn(N, P, L) ;
J   = randn(N, N, P, L) ;
Lam = randn(N, P, L) ;
M   = randn(N, N, P, L) ;
W   = randn(D, N, L) ;
b   = randn(N, L) ;


% TRAIN NETWORK
for iter = 1:20

	% FORWARD STEP
	% initialize
	H(:, :, 1) = W(:, :, 1)' * feval(phi{1}, X) + b(:, 1) ;
	for p = 1:P,  J(:, :, p, 1) = W(:, :, 1)' * Jacobian(phi{1}, param{1}, X(:, p)) * eye(N) ; end
	% propagate
	for l = 2:L
		for p = 1:P
			% on hidden states
			H(:, p, l) =  W(:, :, l)' * feval(phi{l}, H(:, p, l-1)) + b(:, l) ;
			% on jacobians
			J(:, :, p, l) = W(:, :, l)' * Jacobian(phi{l}, param{l}, H(:, p, l-1)) * J(:, :, p, l-1) ;
		end
    end


	% BACKWARD STEP
	% initialize
	Lam(:, :, L) = gam * (X - H(:, :, L)) ;
	for p = 1:P, M(:, :, p, L) = -bet * J(:, :, p, L); end
	% propagate
	for l = L-1:-1:1
		for p = 1:P
			% on multipliers of J
			M(:, :, p, l) = Jacobian(phi{l+1}, param{l+1}, H(:, p, l))' * W(:, :, l+1) * M(:, :, p, l+1) ; 
			% on multipliers of H
			s = zeros(size(Lam(:, p, l))) ;
			A = W(:, :, l+1) * M(:, :, p, l+1) ;
			for d = 1:D
				s = s + Hessian(phi{l+1}, param{l+1}, d, H(:, p, l)) * J(:, :, p, l) * A(d, :)' ;
			end
			Lam(:, p, l) = Jacobian(phi{l+1}, param{l+1}, H(:, p, l))' * W(:, :, l+1) * Lam(:, p, l+1) + s ;
		end
	end


	% UPDATE WEIGHTS
	% initialize
	sum_1 = zeros(size(W(:, :, 1))) ;
	sum_2 = zeros(size(W(:, :, 1))) ;
	for p = 1:P
		sum_1 = sum_1 + feval(phi{1}, X(:, p)) * Lam(:, p, 1)' ;
		sum_2 = sum_2 + Jacobian(phi{1}, param{1}, X(:, p)) * M(:, :, p, 1)' ;
	end
	W(:, :, 1) = 1/eta * (sum_1 + sum_2) ;
	b(:, 1) = mean(H(:, :, 1), 2) ;
	% propagate
	for l = 2:L
		% update W
		sum_1 = zeros(size(W(:, :, l))) ;
		sum_2 = zeros(size(W(:, :, l))) ;
		for p = 1:P
			sum_1 = sum_1 + feval(phi{l}, H(:, p, l-1)) * Lam(:, p, l)' ;
			sum_2 = sum_2 + Jacobian(phi{l}, param{l}, H(:, p, l-1)) * J(:, :, p, l-1) * M(:, :, p, l)' ;
		end
		W(:, :, l) = 1/eta * (sum_1 + sum_2) ;
		% update b
		b(:, l) = mean(H(:, :, l), 2) ;
	end


	% CHECK FOR CONVERGENCE
    abs(sum(sum(sum(Lam, 2), 3), 1))
    if abs(sum(sum(sum(Lam, 2), 3), 1)) < 1e-3
		disp( "stopped at iteration " + num2str(iter) ) ;
		break
	end

end


% DISPLAY RESULTS
x = -5:0.1:5 ;
f = x ;
for l = 1:L
	f = W(:, :, l)' * feval(phi{l}, f) + b(:, l) ;
end

figure()
box on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
plot(x, f, 'color', [0 0 1], 'linewidth', 1)
plot(X, X, 'rx')
hold off
%title(strcat('eta = ', num2str(eta),', gama = ', num2str(gamma) ))


%%
% FUNCTIONS

% function p = phi(fun, param, pattern)
% % evaluate feature map in pattern
% 
% 	% extract useful parameters
% 	N = size(pattern, 1) ;
% 
% 	fun = lower(fun);
% 	switch fun
% 		case {'tanh'}
% 			p = tanh(pattern) ;
% 
% 		case {'poly', 'polynomial'}
%             %p = 
% 			
% 		otherwise 
% 			error('feature map not recognized') ;
% 	end
% 
% end

function J = Jacobian(fun, param, pattern)
% evaluate jacobian of phi in specific pattern

	% extract useful parameters
	N = size(pattern, 1) ;

	fun = lower(fun);
	switch fun
		case {'tanh'}
			%J = diag( 1./ cosh( pattern ).^2) ;
			J = diag( 1 - tanh( pattern ).^2) ;

		case {'poly', 'polynomial'}
			
		otherwise 
			error('feature map not recognized') ;
	end

end


function H = Hessian(fun, param, d, pattern)
% evaluate hessian of phi_d in specific pattern

	% extract useful parameters
	N = size(pattern, 1) ;

	fun = lower(fun);
	switch fun
		case {'tanh'}
			H = zeros(N, N) ;
			H(d, d) = -2*tanh(pattern(d)) * (1-tanh(pattern(d)).^2) ;
			
		case {'poly', 'polynomial'}
			
		otherwise 
			error('feature map not recognized') ;
	end

end