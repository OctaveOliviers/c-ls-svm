% @Author: OctaveOliviers
% @Date:   2020-03-05 09:54:32
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 11:17:30

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


		% compute energy in state X
		function [E, varargout] = energy(obj, X, varargin)
			% X 	states to compute energy, error and eigenvalues for in columns

			X_new = obj.simulate_one_step( X ) ;

			e_kin = vecnorm( X-X_new, 2, 1 ) ;

			switch obj.space
				case { "primal", "p" }
					e_pot = zeros(1, size(X, 2)) ;
					for i = 1:size(X, 2)
						F = jac( X(:, i), obj.phi, obj.theta ) ;
						e_pot(i) = trace( obj.W' * F * F' * obj.W ) ;
					end					

					if (nargout>2)
						eig_jac = zeros( size(X) );
						for i = 1:size(X, 2)
							F = jac( X(:, i), obj.phi, obj.theta ) ;
							eig_jac(:, i) = eig( obj.W' * F ) ;
						end	
						
					end

				case { "dual", "d" }
					e_pot = zeros(1, size(X, 2)) ;
					for i = 1:size(X, 2)
						PTj = phiTjac( obj.patterns, X(:, i), obj.phi, obj.theta ) ;
						jTP = jacTphi( X(:, i), obj.patterns, obj.phi, obj.theta ) ;
						JTj = jacTjac( obj.patterns, X(:, i), obj.phi, obj.theta ) ;
						jTJ = jacTjac( X(:, i), obj.patterns, obj.phi, obj.theta ) ;

						e_pot(i) = trace( 1/obj.p_reg^2 * (obj.L_e*PTj + obj.L_d*JTj) * (jTP*obj.L_e' + jTJ*obj.L_d') ) ;
					end

					if (nargout>2)
						eig_jac = zeros( size(X) );
						for i = 1:size(X, 2)
							PTj = phiTjac( obj.patterns, X(:, i), obj.phi, obj.theta ) ;
							jTP = jacTphi( X(:, i), obj.patterns, obj.phi, obj.theta ) ;
							JTj = jacTjac( obj.patterns, X(:, i), obj.phi, obj.theta ) ;
							jTJ = jacTjac( X(:, i), obj.patterns, obj.phi, obj.theta ) ;
							
							eig_jac(:, i) = eig( 1/obj.p_reg * (obj.L_e*PTj + obj.L_d*JTj) ) ;
						end	
						
					end
			end

			E = obj.p_err * e_kin.^2 + obj.p_drv * e_pot ;

			if (nargout>1)
				varargout{1} = e_kin ;
				varargout{2} = eig_jac ;
			end
		end
	end
end