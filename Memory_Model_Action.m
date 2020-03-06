% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:57
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 09:18:42

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
	end
end