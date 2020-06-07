% Created  by OctaveOliviers
%          on 2020-03-29 19:31:12
%
% Modified on 2020-06-04 13:13:18

classdef Hopfield_Network < CLSSVM

    properties
        % model architecture
        phi     % activation function as string
    end

    methods
        % constructor
        function obj = Hopfield_Network(phi)
            % superclass constructor
            obj@CLSSVM() ;
            % architecture
            obj.phi = phi ;     % string
            % model information
            obj.name    = "Hopfield Network" ;
        end


        % train model
        function obj = train(obj, X)
            % X         patterns to memorize in columns
            
            % extract useful parameters
            [N, ~]  = size(X) ;
            obj.X   = X ;

            % center patterns around origin
            obj.b   = mean( X, 2 ) ;
            X_c     = X - obj.b ;

            % model parameters
            obj.W   = 1/N * X_c*X_c' ;

            disp("Hopfield model trained")
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            f = feval( obj.phi, obj.W' * (x - obj.b) ) ;
        end


        % compute energy in state X
        function [E] = energy(obj, X)
            % X     states to compute energy, error and eigenvalues for in columns

            % extract usefull information
            [N, P] = size(X) ;

            E = diag( X' * obj.W * (X-obj.b) ) ;
        end
    end
end