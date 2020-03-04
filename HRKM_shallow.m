% Octave Oliviers - March 4th 2020
clear all
clc

rng(5) ;

dim_patterns = 1 ;
num_patterns = 5 ;

% create patterns to memorize
patterns = 2*randn(dim_patterns, num_patterns) ;


formulation = 'primal' ;
feature_map = 'tanh' ;
parameter   = 0.5 ;

% build model to memorize patterns
p_err  = 1e4 ; % importance of error
p_reg  = 1e1; % importance of regularization
p_drv  = 1e2 ; % importance of minimizing derivative

window = 6 ;
x = -window:0.1:window ;

model = Memory_Model(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;

f = model.simulate(x) ;

figure()
box on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
plot(x, f, 'color', [0 0 1], 'linewidth', 1)
plot(patterns, patterns, 'rx')
hold off
title(strcat('p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg)',', p_drv = ', num2str(p_drv) ))
