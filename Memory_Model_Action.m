% @Author: OctaveOliviers
% @Date:   2020-03-05 10:01:57
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 10:21:50

classdef Memory_Model_Action < Memory_Model
	
	properties
		num_lay		% number of layers
		models		% cell of shallow models that represent each step of the action
	end

	methods
		% constructor
		function obj = Memory_Model_Action(num_lay, space, phi, theta, p_err, p_drv, p_reg)

			obj.num_lay	= num_lay ;
			obj.models	= cell(num_lay) ;

			% architecture
			obj.space 	= space ;
			obj.phi 	= phi ;
			obj.theta 	= theta ;
			% hyper-parameters
			obj.p_err 	= p_err ;
			obj.p_drv 	= p_drv ;
			obj.p_reg 	= p_reg ;
			% shallow model for each step of the action
			for l = 1:num_lay
				obj.models{l} = Memory_Model_Shallow() ;
			end

		end

		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(JJ^T) + p_reg/2*Tr(W^TW)
		function obj = train(obj, data)
			% data 		patterns to memorize
			
			% check for correctness of input
			assert( ndims(data)==3 ) ;
			assert( size(data, 3)==obj.num_lay ) ;

			% add and train each model
			models(end+1)

		end
	end
end