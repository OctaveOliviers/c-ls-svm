% Created  by OctaveOliviers
%          on 2020-04-11 14:59:10
%
% Modified on 2020-04-27 22:30:11

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
            if nargin == 6
                obj.theta = varargin{1} ; % float
            end
        end


        % store layer error, jacobian and Lagrange function
        function obj = store_lagrange_param(obj)
            obj.E = obj.layer_error( obj.X, obj.Y ) ;
            obj.J = obj.layer_jacobian( obj.X ) ;
            obj.L = obj.layer_lagrangian( obj.X, obj.Y ) ;
        end
    end
end