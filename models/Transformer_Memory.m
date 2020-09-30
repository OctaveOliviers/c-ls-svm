% Created  by OctaveOliviers
%          on 2020-09-25 17:15:56
%
% Modified on 2020-09-30 13:38:48

classdef Transformer_Memory

    properties
        X        % memories
        E        % error on the memories
        param    % temperature of softmax function
        name     % name of model
    end

    methods
        % constructor
        function obj = Transformer_Memory( memories, param )

            obj.X       = memories ;
            obj.param   = param ;
            obj.E       = obj.model_error( ) ;

            obj.name    = "Transformer Memory" ;            
        end


        % getter function
        function N = dim_memories(obj)

            N = size(obj.X, 1) ;

        end


        % getter function
        function P = num_memories(obj)

            P = size(obj.X, 2) ;

        end


        % compute the softmax weights in state x
        function p = softmax_weights(obj, x)
            % x         vector of state to compute weights

            p = softmax( obj.param * obj.X' * x ) ;

        end


        % update the state
        function x_kpo = simulate_one_step(obj, x_k)
            % x_k     vector of current state to update

            x_kpo = obj.X * obj.softmax_weights(x_k) ;

        end


        % compute the error on a memory
        function E = model_error(obj, varargin)
            % varargin  
            %       index of selected memories

            if nargin == 1
                % compute error in all the memories
                E = obj.X - obj.simulate_one_step(obj.X) ;

            elseif nargin==2
                % compute error in selected memories
                idx = varargin{1} ;
                E = obj.X(:,idx) - obj.simulate_one_step( obj.X(:,idx) ) ;

            else
                error("Not correct number of arguments")
            end
        end


        % compute the Jacobian in a memory
        function J = model_jacobian_3D(obj, varargin)
            % varargin  
            %       index of selected memories

            if nargin == 1
                % compute jacobian in all the memories
                [N, P] = size( obj.X ) ;
                J = NaN(N, N, P) ;
                P = obj.softmax_weights(obj.X) ;
                for i = 1:size(P,2)
                    J(:,:,i) = obj.param * ( obj.X * ( diag(P(:,i)) - P(:,i)*P(:,i)' ) * obj.X' ) ;
                end

            elseif nargin==2
                % compute jacobian in selected memories
                idx = varargin{1} ;
                [N, P] = size( obj.X(:,idx) ) ;
                J = NaN(N, N, P) ;
                P = obj.softmax_weights(obj.X(:,idx)) ;
                for i = 1:size(P,2)
                    J(:,:,i) = obj.param * ( obj.X * ( diag(P(:,i)) - P(:,i)*P(:,i)' ) * obj.X' ) ;
                end
            else
                error("Not correct number of arguments")
            end
        end
    end
end