% @Author: OctaveOliviers
% @Date:   2020-03-05 09:51:23
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 22:14:43

classdef Memory_Model_Shallow < Memory_Model
	
	properties

	end

	methods
		% constructor
		function obj = Memory_Model_Shallow(space, phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj 		= obj@Memory_Model(space, phi, theta, p_err, p_drv, p_reg) ;
			% subclass specific variables
			obj.num_lay	= 1 ;
		end


		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(JJ^T) + p_reg/2*Tr(W^TW)
		function obj = train(obj, X, varargin)
			% X 		patterns to memorize
			% varargin	contains Y to map patterns X to (for stacked architectures)
			
			% extract useful parameters
			[N, P] 			= size(X) ;
			obj.patterns 	= X ;

			if ( size(varargin)==0 )
				Y = X ;
			else
				Y = varargin{1} ;
			end

			switch obj.space

				case {'primal', 'p'}
					% feature map in each data point
					f = feval(obj.phi, X) ;
					% jacobians of feature map in each data points
					F = jacobian_matrix(X, obj.phi, obj.theta) ;
					% dimension of dual space
					D = size(f, 1) ;

					% matrices for linear system AX=B
					A = zeros( D+1, D+1 ) ;
					B = zeros( D+1, N ) ;

					% left-hand side
					A( 1:D, 1:D ) = f*f' + obj.p_reg*F*F'/obj.p_err ;
					A( 1:D, end ) = sum(f, 2) ;
					A( end, 1:D ) = sum(f, 2) ;
					A( end, end ) = P ;

					% right-hand side
					B( 1:D, : ) = f*Y' ;
					B( end, : ) = sum(Y, 2) ;

					% compute parameters
					v = A\B ;
					%
					obj.W = v(1:N, :) ;
					obj.b = v(end, :)' ;
		            
		        case {'dual', 'd'}
					% build kernel terms
					pTp = phiTphi(X, X, obj.phi, obj.theta) ;
					pTj = phiTjac(X, X, obj.phi, obj.theta) ;
					jTp = jacTphi(X, X, obj.phi, obj.theta) ;
					jTj = jacTjac(X, X, obj.phi, obj.theta) ;
					    
					% % matrices for linear system AX=B
					A = zeros( P+P*N+1, P+P*N+1 ) ;
					B = zeros( P+P*N+1, N ) ;

					% left-hand side
					A(1:P,       1:P)       = pTp/obj.p_reg + eye(P)/obj.p_err ;
					A(1:P,       P+1:end-1) = pTj/obj.p_reg ;
					A(P+1:end-1, 1:P)       = jTp/obj.p_reg ;
					A(P+1:end-1, P+1:end-1) = jTj/obj.p_reg + eye(P*N)/obj.p_drv ;
					A(1:P, 		 end)       = 1 ;
					A(end,       1:P)       = 1 ;
					    
					% right-hand side
					B(1:P, :) = Y' ;

					% compute parameters
					v = A\B ;
					%
					obj.L_e	= v(1:P, :)' ;
					obj.L_d	= v(P+1:end-1, :)' ;
					obj.b 	= v(end, :)' ;
		    end
		    disp("model trained")
		end


		% visualize dynamical model
		function visualize(obj, varargin)

		    % can only visualize 1D and 2D data
		    assert( size(obj.patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

		    % extract useful information
		    dim_data = size(obj.patterns, 1) ;
		    num_data = size(obj.patterns, 2) ;

		    % if data is one dimensional, visualize update function
		    if (dim_data==1)
		        
		        figure('position', [300, 500, 600, 500])
		        
		        x = 1.5*min(obj.patterns, [], 'all') : ...
			    	(max(obj.patterns, [], 'all')-min(obj.patterns, [], 'all'))/10/num_data : ...
			  		1.5*max(obj.patterns, [], 'all') ;

	            box on
	            hold on

	            % identity map
	            plot(x, x, 'color', [0 0 0])

		        % update function f(x_k)
	            f = obj.simulate_one_step(x) ;
	            plot(x, f, 'color', [0 0 1], 'linewidth', 1)

	            % simulate model from initial conditions in varargin
				if (nargin>1)
					x_k = varargin{1} ; 
					p 	= obj.simulate( x_k ) ;
					
					P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
					P(:, :, 1:2:end) = p ;
					P(:, :, 2:2:end) = p ;

					for i = 1:size(P, 2)
						plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'color', [0 0 0], 'linewidth', 1)
					end
					plot(p(:, :, 1), p(:, :, 1), 'kx')
				end

				% patterns to memorize
				plot(obj.patterns, obj.patterns, 'rx')

	            hold off
	            xlabel('x_k')
	            ylabel('x_{k+1}')
	            % axes through origin
	            axis equal
	            ax = gca;
				ax.XAxisLocation = 'origin';
				ax.YAxisLocation = 'origin';
		        title( join([ 'p_err = ', num2str(obj.p_err), ...
		        			', p_reg = ', num2str(obj.p_reg), ...
		        			', p_drv = ', num2str(obj.p_drv) ]))
		    
		    % if data is 2 dimensional, visualize vector field with nullclines
			elseif (dim_data==2)
		    
				figure('position', [300, 500, 600, 500])

				box on
	            hold on

	            % nullclines
				wdw = 8 ; % window
				prec = wdw/20 ;
				x = -wdw:prec:wdw ;
				y = -wdw:prec:wdw ;
				[X, Y] = meshgrid(x, y) ;
				%
				F = obj.simulate_one_step( [ X(:)' ; Y(:)' ] ) ; % maybe this is just simulate_one_step
				f1 = reshape(F(1, :), [length(x), length(y)]) ;
				f2 = reshape(F(2, :), [length(x), length(y)]) ;
				%
				c1 = contour(x,y,X-f1,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', '--') ;
				c2 = contour(x,y,Y-f2,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5]) ;

	            % simulate model from initial conditions in varargin
				if (nargin>1)
					x_k = varargin{1} ; 
					p 	= obj.simulate( x_k ) ;
					
					for i = 1:size(p, 2)
						plot(squeeze(p(1, i, :)), squeeze(p(2, i, :)), 'color', [0 0 0], 'linewidth', 1)
					end
					plot(p(1, :, 1), p(2, :, 1), 'ko')
				end

				% patterns to memorize
				plot(obj.patterns(1, :), obj.patterns(2, :), 'rx')

	            hold off
	            xlabel('x_1')
	            ylabel('x_2')
	            % axes through origin
	            % axis equal
		        title( join([ 'p_err = ', num2str(obj.p_err), ...
		        			', p_reg = ', num2str(obj.p_reg), ...
		        			', p_drv = ', num2str(obj.p_drv) ]))

		        legend('x_1 nullcline', 'x_2 nullcline') ;


		    end
		end


		% simulate model over one step
		function f = simulate_one_step(obj, x)
			% x		matrix with start positions to simulate from as columns

			switch obj.space

				case {"primal", "p"}
		            f = obj.W' * feval(obj.phi, x) + obj.b ;

		        case {"dual", "d"}
					pTp = phiTphi(obj.patterns, x, obj.phi, obj.theta) ;
					jTp = jacTphi(obj.patterns, x, obj.phi, obj.theta) ;

					f   = (obj.L_e*pTp + obj.L_d*jTp)/obj.p_reg + obj.b ;
		    end
		end
	end
end