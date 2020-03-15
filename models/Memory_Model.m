% @Author: OctaveOliviers
% @Date:   2020-03-05 09:54:32
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 22:30:45

classdef Memory_Model

	properties
		patterns
		% model architecture
		space 		% 'primal' or 'dual'
		phi			% feature map as string
		theta		% parameter of feature map
		num_lay		% number of layers
		% model hyper-parameters
		p_err		% importance of minimizing error
		p_drv		% importance of minimizing derivative
		p_reg		% importance of regularization
		% model parameters
		W 			% primal weights
		L_e			% dual Lagrange parameters for error
		L_d			% dual Lagrange parameters for derivative
		b		%	 bias
	end

	methods
		% constructor
		function obj = Memory_Model(phi, theta, p_err, p_drv, p_reg)
			% architecture
			% obj.space 	= space ;	% string
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


		% compute energy in state X
		function [E, varargout] = energy(obj, X)
			% X 	states to compute energy, error and eigenvalues for in columns

			switch obj.phi
				case { 'tanh' }
					E = 1/2 * (vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*log(cosh(X)) - 2*obj.b'*X ) ;
				
				case { 'sign' }
					E = 1/2 * (vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*abs(X) - 2*obj.b'*X ) ;

				otherwise
					warning('not exact formulation for energy yet');

					% error term
					err   = obj.model_error( X ) ;
					e_kin = vecnorm( err, 2, 1 ).^2 ;

					% derivative term
					e_pot = zeros(1, size(X, 2)) ;
					for p = 1:size(X, 2)
						J = obj.model_jacobian( X(:, p) ) ;
						e_pot(p) = trace( J'*J ) ;
					end					

					E = obj.p_err/2 * e_kin + obj.p_drv/2 * e_pot ;
			end


			if (nargout>2)
				eig_jac = zeros( size(X) );
				for p = 1:size(X, 2)
					eig_jac(:, p) = eig( -obj.model_jacobian( X(:, p) ) ) ;
				end
			end			

			if (nargout>1)
				varargout{1} = vecnorm( obj.model_error( X ), 2, 1 ) ;
				varargout{2} = eig_jac ;
			end
		end
	end
end