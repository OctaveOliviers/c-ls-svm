% Created  by OctaveOliviers
%          on 2020-04-14 17:40:10
%
% Modified on 2020-04-27 22:07:06

clc
clear all


N = 10 ; 
P = 100 ; 

eta = N; 
lam = 10 ; 

X = randn(N, P) ; 

M = sign(X)' * sign(X) / eta + eye(P)/lam ;

figure ; imagesc( M ) ; colorbar ; 

