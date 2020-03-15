% @Author: OctaveOliviers
% @Date:   2020-03-05 09:51:23
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 17:31:46

classdef Memory_Model_Shallow < Memory_Model

	methods
		% constructor
		function obj = Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj@Memory_Model(phi, theta, p_err, p_drv, p_reg)
			% subclass specific variables
			obj.num_lay	= 1 ;
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
		        
		        figure('position', [300, 500, 600, 500])
		        
		        x = 1.5*min(obj.patterns, [], 'all') : ...
			    	(max(obj.patterns, [], 'all')-min(obj.patterns, [], 'all'))/20/num_data : ...
			  		1.5*max(obj.patterns, [], 'all') ;

	            box on
	            hold on

	            % identity map
	            yyaxis left
	            ylabel('x_{k+1}')
	            plot(x, x, 'color', [0 0 0])

		        % update function f(x_k)
	            f = obj.simulate_one_step(x) ;
	            plot(x, f, 'b-', 'linewidth', 1)

	            % simulate model from initial conditions in varargin
				if (nargin>1)
					x_k = varargin{1} ; 
					p 	= obj.simulate( x_k ) ;
					
					P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
					P(:, :, 1:2:end) = p ;
					P(:, :, 2:2:end) = p ;

					for i = 1:size(P, 2)
						plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'k-', 'linewidth', 0.5)
					end
					plot(p(:, :, 1), p(:, :, 1), 'kx')
				end

				% patterns to memorize
				plot(obj.patterns, obj.patterns, 'rx')

				yyaxis right
				ylabel('Energy E(x_{k})')
				% energy surface
				E = obj.energy( x ) ;
				plot(x, E, 'g-') ;

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
		    
		    % if data is 2 dimensional, visualize vector field with nullclines
			elseif (dim_data==2)
		    
				figure('position', [300, 500, 600, 500])

				box on
	            hold on

	            % patterns to memorize
				plot(obj.patterns(1, :), obj.patterns(2, :), 'rx', 'linewidth', 2)

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
				E = obj.energy( [ X(:)' ; Y(:)' ] ) ;
				E = reshape( E, [length(x), length(y)]) ;
				%
				% test for energy
				% e = sum( phiTphi([ X(:)' ; Y(:)' ], obj.patterns, obj.phi, obj.theta), 2 ) ;
				% E = reshape(e, [length(x), length(y)]) ;
				%
				contour(x, y, X-f1,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', '--') ;
				contour(x, y, Y-f2,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', ':') ;
				contour(x, y, E) ;

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
				plot(obj.patterns(1, :), obj.patterns(2, :), 'rx', 'linewidth', 2)

				hold off
				xlabel('x_1')
				ylabel('x_2')
				xlim([-wdw, wdw])
				ylim([-wdw, wdw])
				% axes through origin
				% axis equal
				title( join([ 'p_err = ', num2str(obj.p_err), ...
							', p_reg = ', num2str(obj.p_reg), ...
							', p_drv = ', num2str(obj.p_drv) ]))

				legend('pattern', 'x_1 nullcline', 'x_2 nullcline') ;

		    end
		end
	end
end