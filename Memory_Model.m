% @Author: OctaveOliviers
% @Date:   2020-03-05 09:54:32
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 10:29:56

classdef Memory_Model

	properties
		patterns
		% model architecture
		space 	% 'primal' or 'dual'
		phi		% feature map as string
		theta	% parameter of feature map
		% model hyper-parameters
		p_err	% importance of minimizing error
		p_drv	% importance of minimizing derivative
		p_reg	% importance of regularization
		% model parameters
		W 		% primal weights
		L_e		% dual Lagrange parameters for error
		L_d		% dual Lagrange parameters for derivative
		b		% bias
	end

	methods
		% constructor
		function obj = Memory_Model(space, phi, theta, p_err, p_drv, p_reg)
			
		end

		% visualize model
		function visualize

		end

		% simulate model
		function x_new = simulate(obj, x)
			% x		matrix with start positions to simulate from as columns

			% initialize variables
			x_old = x ;
			x_new = simulate_one_step(obj, x_old) ;
			% update state untill it has converged
			while (norm(x_old-x_new) <= 1e-10)
				x_old = x_new ;
				x_new = simulate_one_step(obj, x_old) ;
			end
		end
	end
end