% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 10:30:19

clear all
clc

rng(5) ;

dim_patterns = 1 ;
num_patterns = 5 ;

% create patterns to memorize
patterns = [0.1*randn(dim_patterns, num_patterns)+0, ...
			0.1*randn(dim_patterns, num_patterns)+2, ...
			0.1*randn(dim_patterns, num_patterns)+4 ] ;

% model architecture
formulation = 'dual' ;
feature_map = 'gauss' ;
parameter   = 0.8 ;
num_layers	= 1 ;

% build model to memorize patterns
p_err  = 1e4 ;	% importance of error
p_reg  = 1e1 ;	% importance of regularization
p_drv  = 1e3 ;	% importance of minimizing derivative

window = 10 ;
x = -window:0.1:window ;

model = Memory_Model_Shallow(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;

f = model.simulate_one_step(x) ;

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
