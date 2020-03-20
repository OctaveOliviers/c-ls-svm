% @Author: OctaveOliviers
% @Date:   2020-03-19 16:58:27
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-19 16:58:39


classdef Hopfield_network

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
			while (norm(x_old-x_new) >= 1e-3)
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

			% extract usefull information
			[N, P] = size(X) ;

			switch obj.phi
				case { 'tanh' }
					E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*log(cosh(X)) - 2*obj.b'*X ) ;

				
				case { 'sign' }
					E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*abs(X) - 2*obj.b'*X ) ;

				case { 'polynomial', 'poly', 'p' }
					if ( N==1 )
						int_k 	= obj.L_e * ( ( obj.patterns'*X + obj.theta(2) ).^(obj.theta(1)+1) ./ obj.patterns' ) / (obj.theta(1)+1) ;
						k 		= obj.L_d * ( phiTphi( obj.patterns, X, obj.phi, obj.theta ) .* X ./ obj.patterns' ...
											- phiTphi( obj.patterns, X, obj.phi, [obj.theta(1)+1, obj.theta(2)] ) ./ (obj.patterns.^2)' / (obj.theta(1)+1) ) ;

						E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2/obj.p_reg * (int_k + k) - 2*obj.b'*X ) ;
					else
						warning('do not know energy formulation yet')
						E = zeros(1, P) ;
					end

				case { 'gaussian', 'gauss', 'g' }
					if ( N==1 )
						int_k 	= obj.theta*sqrt(pi/2) * obj.L_e * erfc( (obj.patterns' - X) / (sqrt(2)*obj.theta) ) ;
						k 		= - obj.L_d * phiTphi( obj.patterns, X, obj.phi, obj.theta ) ;

						E = ( 1/2 * vecnorm(X, 2, 1).^2 - 1/obj.p_reg * (int_k + k) - obj.b'*X ) ;
					else
						warning('do not know energy formulation yet')
						E = zeros(1, P) ;
					end

				otherwise
					warning('not exact formulation for energy yet');

					% error term
					err   = obj.model_error( X ) ;
					e_kin = vecnorm( err, 2, 1 ).^2 ;

					% derivative term
					e_pot = zeros(1, P) ;
					for p = 1:P
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



		% visualize dynamical model
		function visualize(obj, varargin)
			% varargin		(1) start positions to simulate model from

		    % can only visualize 1D and 2D data
		    assert( size(obj.patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

		    % extract useful information
		    dim_data = size(obj.patterns, 1) ;
		    num_data = size(obj.patterns, 2) ;

		    % if data is one dimensional, visualize update function
		    if (dim_data==1)
		        
		        figure('position', [300, 500, 400, 300])
		        
		        x = 1.5*min(obj.patterns, [], 'all') : ...
			    	(max(obj.patterns, [], 'all')-min(obj.patterns, [], 'all'))/20/num_data : ...
			  		1.5*max(obj.patterns, [], 'all') ;

	            box on
	            hold on
	            yyaxis left
	            
	            % patterns to memorize
				l_patterns = plot( obj.patterns, obj.patterns, 'rx', 'linewidth', 2 ) ;

	            % update function f(x_k)
	            f = obj.simulate_one_step(x) ;
	            l_update = plot( x, f, 'linestyle', '-', 'color', [0, 0.4470, 0.7410], 'linewidth', 1) ;

	            % simulate model from initial conditions in varargin
				if (nargin>1)
					x_k = varargin{1} ; 
					p 	= obj.simulate( x_k ) ;
					
					P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
					P(:, :, 1:2:end) = p ;
					P(:, :, 2:2:end) = p ;

					for i = 1:size(P, 2)
						plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'k-', 'linewidth', 0.5) ;
					end
					plot(p(:, :, 1), p(:, :, 1), 'kx') ;
				end

				% identity map
	            ylabel('x_{k+1}')
	            l_identity = plot(x, x, 'color', [0.4 0.4 0.4]) ;

				yyaxis right
				% energy surface
				E = obj.energy( x ) ;
				l_energy = plot(x, E, 'linestyle', '-.', 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 1) ;
				ylabel('Energy E(x_{k})')

	            hold off
	            xlabel('x_k')
	            % axes through origin
	            % axis equal
	            ax = gca;
				ax.XAxisLocation = 'origin';
				% ax.YAxisLocation = 'origin';
		        title( join([ 'p_err = ', num2str(obj.p_err), ...
		        			', p_reg = ', num2str(obj.p_reg), ...
		        			', p_drv = ', num2str(obj.p_drv) ]))
		        % title('Polynomial kernel (d=5, t=1)')
		        legend( [l_patterns, l_update, l_energy, l_identity ], {'Pattern', 'Update equation', 'Energy', 'Identity map'} , 'location', 'northwest')

		    % if data is 2 dimensional, visualize vector field with nullclines
			elseif (dim_data==2)
		    
				figure('position', [300, 500, 330, 300])

				box on
	            hold on

	            % patterns to memorize
				l_patterns = plot(obj.patterns(1, :), obj.patterns(2, :), 'rx', 'linewidth', 2) ;

				% energy surface and nullclines
				wdw = 10 ; % window
				prec = wdw/20 ;
				x = -wdw:prec:wdw ;
				y = -wdw:prec:wdw ;
				[X, Y] = meshgrid(x, y) ;			
				%
				F = obj.simulate_one_step( [ X(:)' ; Y(:)' ] ) ;
				f1 = reshape( F(1, :), [length(x), length(y)] ) ;
				f2 = reshape( F(2, :), [length(x), length(y)] ) ;
				% E = obj.energy( [ X(:)' ; Y(:)' ] ) ;
				% E = reshape( E, [length(x), length(y)]) ;
				%
				[~, l_nc1] = contour(x, y, X-f1,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', '--') ;
				[~, l_nc2] = contour(x, y, Y-f2,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', ':') ;
				% contour(x, y, E) ;

				% simulate model from initial conditions in varargin
				if (nargin>1)
					x_k = varargin{1} ; 
					p 	= obj.simulate( x_k ) ;
					
					for i = 1:size(p, 2)
						plot(squeeze(p(1, i, :)), squeeze(p(2, i, :)), 'color', [0 0 0], 'linewidth', 1)
					end
					plot(p(1, :, 1), p(2, :, 1), 'ko')
				end

				hold off
				xlabel('x_1')
				ylabel('x_2')
				xlim([-wdw, wdw])
				ylim([-wdw, wdw])
				% axes through origin
				axis equal
				% title( join([ 'p_err = ', num2str(obj.p_err), ...
				% 			', p_reg = ', num2str(obj.p_reg), ...
				% 			', p_drv = ', num2str(obj.p_drv) ]))
				% title( join([ num2str(obj.num_lay), ' layers ', obj.phi{1} ]) )
				title('5 layers of poly (d=3, t=1)')
				% legend( [l_patterns, l_nc1, l_nc2], {'Pattern', 'x_1 nullcline', 'x_2 nullcline'}, 'location', 'southwest') ;

		    end
		end
	end
end