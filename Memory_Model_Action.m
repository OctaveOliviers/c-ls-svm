% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:57
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 10:42:16

classdef Memory_Model_Action < Memory_Model
	
	properties
		models		% cell of shallow models that represent each step of the action
	end

	methods
		% constructor
		function obj = Memory_Model_Action(num_lay, space, phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj = obj@Memory_Model(space, phi, theta, p_err, p_drv, p_reg) ;
			% subclass specific variables
			obj.num_lay	= num_lay ;
			obj.models	= cell(num_lay) ;
			% shallow model for each step of the action
			for l = 1:num_lay
				obj.models{l} = Memory_Model_Shallow(space, phi, theta, p_err, p_drv, p_reg) ;
			end

		end

		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(JJ^T) + p_reg/2*Tr(W^TW)
		function obj = train(obj, movements)
			% movements 	movements to memorize
			
			% check for correctness of input
			assert( ndims(movements)==3 , ...
					'Movement should have three dimensions.' ) ;
			assert( size(movements, 3)==obj.num_lay+1, ...
					'Movement should have same number of steps as the network has number of layers.' ) ;

			% store movements to memorize
			obj.patterns = movements ;

			% train each model
			for l = 1:obj.num_lay
				obj.models{l} = obj.models{l}.train( movements(:, :, l), movements(:, :, l+1) ) ;
			end
		end


		% simulate model over one entire movement
		function [path, varargout] = simulate(obj, start, varargin)
			% start		matrix with start positions to simulate from as columns

			% path from start
			path = zeros([size(start), obj.num_lay+1]) ;
			path(:, :, 1) = start ;

			% simulate consecutive steps of the movement 	
			for l = 1:obj.num_lay
				path(:, :, l+1) = obj.models{l}.simulate_one_step( path(:, :, l) ) ;
			end

			% visualize the update map of the layer
			if (nargin>2)
				x = varargin{1} ;
				f = zeros([size(x), obj.num_lay]) ;

				for l = 1:obj.num_lay		
					f(:, :, l)		= obj.models{l}.simulate_one_step( x ) ;
				end
				
				varargout{1} = f ;
			end
		end


		% visualize dynamical model
		function visualize(obj, varargin)

		    % can only visualize 1D and 2D data
		    assert( size(obj.patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

		    % % extract useful information
		    dim_data = size(obj.models{1}.patterns, 1) ;
		    num_data = size(obj.models{1}.patterns, 2) ;

		    % if data is one dimensional, visualize update function
		    if (dim_data==1)
		        
		        figure('position', [300, 500, 300*obj.num_lay, 300])
			    for l = 1:obj.num_lay

			    	x = 1.5*min(obj.patterns(:, :, l:l+1), [], 'all') : ...
			    		(max(obj.patterns(:, :, l:l+1), [], 'all')-min(obj.patterns(:, :, l:l+1), [], 'all'))/10/num_data : ...
			  			1.5*max(obj.patterns(:, :, l:l+1), [], 'all') ;

			    	subplot(1, obj.num_lay, l)
		            box on
		            hold on

		            % identity map
		            plot(x, x, 'color', [0 0 0])

   		            % update function
		            f = obj.models{l}.simulate_one_step(x) ;
		            plot(x, f, 'color', [0 0 1], 'linewidth', 1)

					% simulate model from initial conditions in varargin
					if (nargin>1)
						x_k 	= varargin{1} ; 
						x_kp1 	= obj.models{l}.simulate_one_step( x_k ) ;
						
						for i = 1:size(x_k, 2)
			                line(   [x_k(:, i), x_kp1(:, i)], ...
			                        [x_k(:, i), x_kp1(1, i)], ...
			                        'color', [0 0 0], 'linewidth', 1 )
			            end
       		            plot(x_k(:, :), x_k(:, :), 'kx')
					end

		            % plot mouvement to memorize
		            plot(obj.patterns(:, :, l), obj.patterns(:, :, l+1), 'rx')
		            
		            hold off
		            % plot layout
		            xlabel('x_k')
		            ylabel('x_{k+1}')
		            % axes through origin
		            ax = gca;
					ax.XAxisLocation = 'origin';
					ax.YAxisLocation = 'origin';
		        end
		        suptitle( join([ 'p_err = ', num2str(obj.p_err), ...
		        				 ', p_reg = ', num2str(obj.p_reg), ...
		        				 ', p_drv = ', num2str(obj.p_drv) ]))
						    	
		    
		    % if data is 2 dimensional, visualize vector field with nullclines
			elseif (dim_data==2)
		    

		    end
		end
	end
end