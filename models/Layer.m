% Created  by OctaveOliviers
%          on 2020-04-11 14:59:10
%
% Modified on 2020-05-12 14:54:31

classdef Layer

    properties
        X        % points to map from
        Y        % points to map to
        N_in     % dimension of X
        N_out    % dimension of Y
        P        % number of points to store
        % layer parameters
        space    % 'primal' or 'dual'
        phi      % feature map as string
        theta    % parameter of feature map
        % layer hyper-parameters
        p_err    % importance of minimizing error
        p_drv    % importance of minimizing derivative
        p_reg    % importance of regularization
        p_mom    % momentum hyper parameter for evaluation
        % layer weights
        W        % primal weights
        b        % bias
        L_e      % dual weights for error
        L_d      % dual weights for derivative
        % results of optimization proces
        E        % error on each pattern
        J        % jacobian in each pattern
        L        % total lagrange function
    end

    methods

        % constructor
        function obj = Layer(N_out, p_err, p_drv, p_reg, phi, varargin)
            % hyper-parameters
            obj.p_err   = p_err ;   % float
            obj.p_drv   = p_drv ;   % float
            obj.p_reg   = p_reg ;   % float
            % layer parameters
            obj.N_out   = N_out ;   % integer
            obj.phi     = phi ;     % string
            if nargin >= 6
                obj.theta = varargin{1} ; % float
            end
            if nargin >= 7
                obj.p_mom = varargin{2} ; % float
            else
                obj.p_mom = 1 ;
            end
             
        end


        % store layer error, jacobian and Lagrange function
        function obj = store_lagrange_param(obj)
            obj.E = obj.layer_error( obj.X, obj.Y ) ;
            obj.J = obj.layer_jacobian( obj.X ) ;
            obj.L = obj.layer_lagrangian( obj.X, obj.Y ) ;
        end


        % error of model E = Y - W' * phi(X) - B
        function E = layer_error(obj, varargin)
            % X     states to compute error in

            % compute error of model
            if (nargin==1)
                E = obj.E ;

            % % compute error for new output target
            % if (nargin==2)
            %     E = varargin{1} - obj.simulate_one_step( obj.X ) ;

            % compute error in new point input-output pair
            elseif ( nargin == 3 )
                E = varargin{2} - obj.simulate_one_step( varargin{1} ) ;
            end
        end


        % compute gradient of Lagrangian with respect to its output evaluated in columns of Y
        function grad = gradient_lagrangian_wrt_output(obj, Y)

            E = Y - obj.simulate_one_step( obj.X ) ;

            % gradient of error
            grad = obj.p_err * E ;

            % gradient of jacobian
            % none
        end


        % chekch if model is trained
        function bool = is_trained(obj)

            % either primal or dual parameters are non empty
            bool =  ( ~isempty(obj.W) && ~isempty(obj.b) ) ...
                 || ( ~isempty(obj.L_e) && ~isempty(obj.L_d) && ~isempty(obj.b) ) ;

        end
    end
end