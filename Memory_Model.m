% @Author: OctaveOliviers
% @Date:   2020-03-05 09:54:32
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-13 19:46:54

classdef Memory_Model

	properties
		patterns
		% model architecture
		space 	% 'primal' or 'dual'
		phi		% feature map as string
		theta	% parameter of feature map
		num_lay	% number of layers
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
			% architecture
			obj.space 	= space ;	% string
			obj.phi 	= phi ;		% string
			obj.theta 	= theta ;	% float
			% hyper-parameters
			obj.p_err 	= p_err ;	% float
			obj.p_drv 	= p_drv ;	% float
			obj.p_reg 	= p_reg ;	% float
		end


		% simulate model
		function [path, varargout] = simulate(obj, start, varargin)
			% start		matrix with start positions to simulate from as columns
			% varargin 	(1) array of starting values to compute update equation

			% variable to store evolution of state
			path = zeros( [size(start), 2]) ;
			path(:, :, 1) = start ;

			% initialize variables
			x_old = start ;
			x_new = simulate_one_step(obj, x_old) ;
			path(:, :, 2) = x_new ;

			% update state untill it has converged
			while (norm(x_old-x_new) >= 1e-8)
				x_old = x_new ;
				x_new = simulate_one_step(obj, x_old) ;
				path(:, :, end+1) = x_new ;

				% if norm(x_new)>10*max(vecnorm(obj.patterns))
				% 	break
				% end
			end

			% visualize the update map f(x) of the layer
			if (nargin>2)
				x = varargin{1} ;
				varargout{1} = obj.simulate_one_step( x ) ; ;
			end
		end
	end
end