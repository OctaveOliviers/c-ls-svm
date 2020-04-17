% Created  by OctaveOliviers
%          on 2020-04-14 17:40:10
%
% Modified on 2020-04-16 17:25:16

clc
clear all


x0 = [10; 4] ;

x_opt = gradient_descent( @objective, @gradient, x0, 30, 10 )

objective( x_opt )
gradient( x_opt )

function obj = objective( x )
    obj = x' * x ;
end 

function grad = gradient( x )
    grad = 2*x ;
end 