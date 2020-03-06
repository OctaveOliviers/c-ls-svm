% @Author: OctaveOliviers
% @Date:   2020-03-05 09:55:31
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 10:02:55

clear all
clc

dim_movements = 1 ;
num_movements = 2 ;
len_movements = 5 ;

% create patterns to memorize
movements = 2*randn(dim_movements, num_movements, len_movements) ;

% model architecture
formulation = 'dual' ;
feature_map = 'gauss' ;
parameter   = 0.5 ;
num_layers	= len_movements-1 ;

% build model to memorize patterns
p_err  = 1e4 ;	% importance of error
p_reg  = 1e1 ;	% importance of regularization
p_drv  = 1e3 ;	% importance of minimizing derivative


model = Memory_Model_Action(num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(movements) ;

model.visualize() ;

% % visualize dynamical system
% window = 5 ;
% x = -window:0.1:window ;

% % path of each data point as well as dynamical system of each layer
% [p, f] = model.simulate(movements(:, 1, 1), x) ;

% figure('position', [300, 500, 300*num_layers, 300])
% for l = 1:num_layers
% 	subplot(1, num_layers, l)
% 	box	on
% 	hold on
% 	plot(x, x, 'color', [0 0 0])
% 	plot(zeros(size(x)), x, 'color', [0 0 0])
% 	plot(x, zeros(size(x)), 'color', [0 0 0])
% 	grid on
% 	plot(x, squeeze(f(:, :, l)), 'color', [0 0 1], 'linewidth', 1)
% 	for i = 1:size(p, 2)
% 		line(	[squeeze(p(1, i, l)), squeeze(p(1, i, l))], ...
% 				[squeeze(p(1, i, l)), squeeze(p(1, i, l+1))], ...
% 				'color', [0 0 0], 'linewidth', 1 )
% 	end
% 	plot(p(:, :, l), p(:, :, l), 'kx')
% 	plot(movements(:, :, l), movements(:, :, l+1), 'rx')
% 	hold off
% 	xlabel('x_k')
% 	ylabel('x_{k+1}')
% 	ax = gca;
% 	ax.XAxisLocation = 'origin';
% 	ax.YAxisLocation = 'origin';
% end
% suptitle( join([ 'p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg),', p_drv = ', num2str(p_drv) ]))