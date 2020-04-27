% Created  by OctaveOliviers
%          on 2020-03-05 09:51:23
%
% Modified on 2020-04-11 15:01:56

classdef Memory_Model_Shallow < Memory_Model

    properties
        % model information
        name = 'Shallow network'
        
    end

    methods
        % constructor
        function obj = Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg)
            % superclass constructor
            obj@Memory_Model()
            
            % subclass specific variables
            obj.num_lay = 1 ;
            % architecture
            obj.phi     = phi ;     % string
            obj.theta   = theta ;   % float
            % hyper-parameters
            obj.p_err   = p_err ;   % float
            obj.p_drv   = p_drv ;   % float
            obj.p_reg   = p_reg ;   % float
        end
    end
end