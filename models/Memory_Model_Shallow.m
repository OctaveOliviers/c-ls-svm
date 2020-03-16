% @Author: OctaveOliviers
% @Date:   2020-03-05 09:51:23
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-16 18:26:44

classdef Memory_Model_Shallow < Memory_Model

	methods
		% constructor
		function obj = Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj@Memory_Model(phi, theta, p_err, p_drv, p_reg)
			% subclass specific variables
			obj.num_lay	= 1 ;
		end

	end
end