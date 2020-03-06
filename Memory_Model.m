% @Author: OctaveOliviers
% @Date:   2020-03-05 09:54:32
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 10:43:15

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

			% variable to store evolution of state
			path = zeros( [size(start), 2]) ;
			path(:, :, 1) = start ;

			% initialize variables
			x_old = start ;
			x_new = simulate_one_step(obj, x_old) ;
			path(:, :, 2) = x_new ;

			% update state untill it has converged
			while (norm(x_old-x_new) >= 1e-5)
				x_old = x_new ;
				x_new = simulate_one_step(obj, x_old) ;
				path(:, :, end+1) = x_new ;
			end

			% visualize the update map f(x) of the layer
			if (nargin>2)
				x = varargin{1} ;
				varargout{1} = obj.simulate_one_step( x ) ; ;
			end
		end


		% visualize dynamical model
		function visualize(obj, varargin)

		    % can only visualize 1D and 2D data
		    assert( size(obj.patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

		    % read input
		    if (nargin>1)
		    	% where to start simulation from
		    	start_sim = varargin{1} ;
		    end

		    % extract useful information
		    dim_data = size(obj.patterns, 1) ;
		    num_data = size(obj.patterns, 2) ;

		    % check if network memorized patterns or movements
		    if ( ndims(obj.patterns)==3 )
		        len_data 	= size(obj.patterns, 3) ;
		        bool_move	= true ;
		    end


		    % if data is one dimensional, visualize update function
		    if (dim_data==1)
		        
		        x = 1.5*min(patterns) : (max(patterns)-min(patterns))/num_data : 1.5*max(patterns) ;
				
				if exist('start_sim')
					[p, f] = model.simulate(startsim, x) ;
				else
					f = model.simulate_one_step(x) ;
				end

				% visaulize memorized static patterns
				if bool_move
					figure('position', [300, 500, 300*obj.num_lay, 300])
			        for l = 1:obj.num_lay
			            subplot(1, obj.num_lay, l)
			            box on
			            hold on
			            % identity map
			            plot(x, x, 'color', [0 0 0])
			            % update function
			            plot(x, squeeze(f(:, :, l)), 'color', [0 0 1], 'linewidth', 1)
			            % simulation from start_sim
			            if exist('p')
			            	for i = 1:size(p, 2)
				                line(   [squeeze(p(1, i, l)), squeeze(p(1, i, l))], ...
				                        [squeeze(p(1, i, l)), squeeze(p(1, i, l+1))], ...
				                        'color', [0 0 0], 'linewidth', 1 )
				            end
	       		            plot(p(:, :, l), p(:, :, l), 'kx')
			            end
			            % patterns/movement to memorize
			            if 
			            plot(obj.(:, :, l), movements(:, :, l+1), 'rx')
			            hold off
			            xlabel('x_k')
			            ylabel('x_{k+1}')
			            % axes through origin
			            ax = gca;
						ax.XAxisLocation = 'origin';
						ax.YAxisLocation = 'origin';
			        end
			        suptitle( join([ 'p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg),', p_drv = ', num2str(p_drv) ]))
				
				% visaulize memorized static patterns
				else
					figure('position', [300, 500, 600, 600])
			        for l = 1:obj.num_lay
			            subplot(1, obj.num_lay, l)
			            box on
			            hold on
			            % identity map
			            plot(x, x, 'color', [0 0 0])
			            % update function
			            plot(x, squeeze(f(:, :, l)), 'color', [0 0 1], 'linewidth', 1)
			            % simulation from start_sim
			            if exist('p')
			            	for i = 1:size(p, 2)
				                line(   [squeeze(p(1, i, l)), squeeze(p(1, i, l))], ...
				                        [squeeze(p(1, i, l)), squeeze(p(1, i, l+1))], ...
				                        'color', [0 0 0], 'linewidth', 1 )
				            end
	       		            plot(p(:, :, l), p(:, :, l), 'kx')
			            end
			            % patterns/movement to memorize
			            if 
			            plot(obj.(:, :, l), movements(:, :, l+1), 'rx')
			            hold off
			            xlabel('x_k')
			            ylabel('x_{k+1}')
			            % axes through origin
			            ax = gca;
						ax.XAxisLocation = 'origin';
						ax.YAxisLocation = 'origin';
			        end
			        suptitle( join([ 'p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg),', p_drv = ', num2str(p_drv) ]))
				end

		    
		    % if data is 2 dimensional, visualize vector field with nullclines
			elseif (dim_data==2)
		    

		    end
		end


	end
end