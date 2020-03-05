% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 14:47:31

clear all
clc

rng(5) ;

dim_patterns = 1 ;
num_patterns = 5 ;

% create patterns to memorize
patterns = [0.1*randn(dim_patterns, num_patterns)-2, ...
			0.1*randn(dim_patterns, num_patterns)+1, ...
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

window = 5 ;
x = -window:0.1:window ;

model = Memory_Model_Shallow(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;

f = model.simulate_one_step(x) ;

p = model.simulate(2) ;
P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
P(:, :, 1:2:end) = p ;
P(:, :, 2:2:end) = p ;

figure('position', [300, 500, 600, 500])
box	on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
grid on
plot(x, f, 'color', [0 0 1], 'linewidth', 1)
for i = 1:size(P, 2)
	plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'color', [0 0 0], 'linewidth', 1)
end
plot(p(:, :, 1), p(:, :, 1), 'kx')
plot(patterns, patterns, 'rx')
hold off
title( join([ 'p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg),', p_drv = ', num2str(p_drv) ]))
xlabel('x_k')
ylabel('x_{k+1}')
axis on