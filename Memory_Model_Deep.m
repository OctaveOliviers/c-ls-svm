% @Author: OctaveOliviers
% @Date:   2020-03-05 19:26:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-14 19:56:34

classdef Memory_Model_Deep < Memory_Model
	
	properties
		models		% cell of shallow models in each layer
	end

	methods
		% constructor
		function obj = Memory_Model_Deep(num_lay, space, phi, theta, p_err, p_drv, p_reg)
			% check correctness of inputs
			assert( size(space) == [num_lay, 1] , 'Number of spaces does not match number of layers' ) ;
			assert( size(phi)   == [num_lay, 1] , 'Number of feature maps does not match number of layers' ) ;
			assert( size(theta) == [num_lay, 1] , 'Number of feature parameters does not match number of layers' ) ;

			% superclass constructor
			obj = obj@Memory_Model(space, phi, theta, p_err, p_drv, p_reg) ;
			% subclass specific variables
			obj.num_lay	= num_lay ;
			obj.models	= cell(num_lay, 1) ;
			% shallow model for each step of the action
			for l = 1:num_lay
				obj.models{l} = Memory_Model_Shallow(space{l}, phi{l}, theta{l} p_err, p_drv, p_reg) ;
			end
		end


		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
		function obj = train(obj, X, varargin)
			% X 		patterns to memorize
			% varargin	contains Y to map patterns X to (for stacked architectures)
			
			% extract useful parameters
			[N, P] 			= size(X) ;
			obj.patterns 	= X ;

			% initialize
			% hidden representations of patterns
			H = repmat( X, 1, 1, obj.num_lay+1 ) ;
			% residual
			r = Inf ;
			while ( r > 1e-5 )

				% train each layer
				for l = 1:obj.num_lay
					obj.models{ l } = obj.models{ l }.train( H(:, :, l), H(:, :, l+1) ) ;
				end

				% update hidden layers
				for l = obj.num_lay-1:-1:1

					switch obj.space{ l }
						case {'primal', 'p'}
							L_e_l 	= obj.models{ l}.L_e ; 
							F_lp1	= jac( H(:, :, l), obj.phi{l+1}, obj.theta{l+1} ) ;
							W_lp1	= obj.models{ l+1 }.W ;
							L_e_lp1	= obj.models{ l+1 }.L_e ;
							L_d_lp1	= obj.models{ l+1 }.L_d ;

							grad 	=


						case {'dual', 'd'}
							warning( 'target prop has not yet been implemented for dual training' ) ;
					end

				end

				r

			end

		    disp("model trained")
		end


		% simulate model over one step
		function F = simulate_one_step(obj, X)
			% x		matrix with start positions to simulate from as columns

			F = X ;
			for l = 1:obj.num_lay
				F = obj.models{ l }.simulate_one_step( F ) ;
			end
		end
	end
end