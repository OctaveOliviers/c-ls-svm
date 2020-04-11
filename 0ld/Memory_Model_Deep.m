% Created  by OctaveOliviers
%          on 2020-04-11 14:57:53
%
% Modified on 2020-04-11 15:45:05
classdef Memory_Model_Deep < Memory_Model
    
    properties
        models      % cell of shallow models in each layer
        num_lay     % number of layers
        max_iter    % maximum number of iterations during training
        alpha       % learning rate for gradient descent in hidden states
    end

    methods
        % constructor
        function obj = Memory_Model_Deep( varargin )
            % construct either from cell of ordered shallow models as
            %   obj = Memory_Model_Deep( models )
            %
            % or from its parameters as
            %   obj = Memory_Model_Deep( num_lay, spaces, phis, thetas, p_err, p_drv, p_reg )

            % superclass constructor
            obj = obj@Memory_Model() ;

            if (nargin==1)
                % subclass specific variables
                obj.num_lay     = length(varargin{1}) ;
                obj.models      = varargin{1} ;
                obj.max_iter    = varargin{end} ;
                obj.alpha       = varargin{end-1} ;
                obj.X           = varargin{1}{1}.X ;
                obj.name        = join([ num2str(obj.num_lay), '-layered network (']) ;
                for l = 1:obj.num_lay
                    obj.name    = append( obj.name, join([obj.models{l}.phi, ', ']) ) ;
                end
                obj.name    = append( obj.name(1:end-2), ')' ) ;

            else
                % check correctness of inputs
                assert( length(varargin{2}) == varargin{1} , 'Number of spaces does not match number of layers' ) ;
                assert( length(varargin{3}) == varargin{1} , 'Number of feature maps does not match number of layers' ) ;
                assert( length(varargin{4}) == varargin{1} , 'Number of feature parameters does not match number of layers' ) ;

                % subclass specific variables
                obj.num_lay     = varargin{1} ;
                obj.models      = cell(varargin{1}, 1) ;
                obj.max_iter    = varargin{end} ;
                obj.alpha       = varargin{end-1} ;
                % shallow model for each layer
                for l = 1:obj.num_lay
                    obj.models{l} = build_model(1, varargin{2}{l}, varargin{3}{l}, varargin{4}{l}, varargin{5}, varargin{6}, varargin{7}) ;
                end
                % model information
                obj.name        = join([ num2str(obj.num_lay), '-layered network (']) ;
                for l = 1:obj.num_lay
                    obj.name    = append( obj.name, join([obj.models{l}.phi, ', ']) ) ;
                end
                obj.name    = append( obj.name(1:end-2), ')' ) ;
            end
        end


        



        
    end
end