%% deep dynamical system for memorizing patterns 
% Octave Oliviers - 23nd February 2020


% DECLARE PARAMETERS
% for the data
N = 1 ; 	% number of neurons = dimension of the data
P = 2 ;		% number of patterns
% for the network
L = 4 ;		% number of layers
eta = 1e2 ;
gam = 1e3 ;	% regularization parameter


% BUILD DATASET
X = randn(N, P)


% TRAIN NETWORK

