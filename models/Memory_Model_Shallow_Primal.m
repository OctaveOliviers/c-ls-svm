% Created  by OctaveOliviers
%          on 2020-03-29 19:04:04
%
% Modified on 2020-03-29 21:15:32

classdef Memory_Model_Shallow_Primal < Memory_Model_Shallow
    
    methods
        % constructor
        function obj = Memory_Model_Shallow_Primal(phi, theta, p_err, p_drv, p_reg)
            % superclass constructor
            obj@Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg) ;
            % subclass secific variable
            obj.space   = 'primal' ;
            % model information
            obj.name    = join([ '1-layered network (', phi, ')']) ;
        end


        % train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
        function obj = train(obj, X, varargin)
            % X         patterns to memorize
            % varargin  contains Y to map patterns X to (for stacked architectures)
            
            if ( nargin<3 )
                Y = X ;
            else
                Y = varargin{1} ;

                % check correctness of input
                assert( size(X, 2)==size(Y, 2),  'Numbr of patterns in X and Y do not match.' ) ;
            end     

            % extract useful parameters
            [Nx, P] = size(X) ;
            [Ny, ~] = size(Y) ;
            obj.X   = X ;
            obj.Y   = Y ;

            % feature map in each data point
            f = feval(obj.phi, X) ;
            % jacobians of feature map in each data point
            F = jac(X, obj.phi, obj.theta) ;
            % dimension of dual space
            D = size(f, 1) ;

            % matrices for linear system AX=B
            A = zeros( D+1, D+1 ) ;
            B = zeros( D+1, Ny ) ;

            % left-hand side
            A( 1:D, 1:D ) = f*f' + obj.p_drv*F*F'/obj.p_err + obj.p_reg*eye(D)/obj.p_err ;
            A( 1:D, end ) = sum(f, 2) ;
            A( end, 1:D ) = sum(f, 2) ;
            A( end, end ) = P ;

            % right-hand side
            B( 1:D, : ) = f*Y' ;
            B( end, : ) = sum(Y, 2) ;

            % compute parameters
            v = A\B ;
            % primal
            obj.W   = v(1:D, :) ;
            obj.b   = v(end, :)' ;
            % dual
            obj.L_e = obj.p_err * ( Y - obj.W' * f - obj.b ) ;
            obj.L_d = obj.p_drv * ( - obj.W' * F ) ;

            % store error and jacobian
            obj.E = obj.error( X, Y ) ;
            obj.J = obj.jacobian( X ) ;

            disp("model trained in primal")
        end


        % compute value of Lagrangian
        function L = lagrangian(obj, varargin)
            
            % compute lagrangian of model
            if ( nargin < 2 )
                L = obj.p_err/2 * trace( obj.E' * obj.E ) + ... % error term
                    obj.p_drv/2 * trace( obj.J' * obj.J ) + ... % derivative term
                    obj.p_reg/2 * trace( obj.W' * obj.W ) ;     % regularization term

            % evaluate lagrangian with new parameters
            else
                X = varargin{1} ;
                Y = varargin{2} ;
                
                E = Y - obj.simulate_one_step( X ) ; 
                J = obj.jacobian( X ) ;

                L = obj.p_err/2 * trace( E' * E ) + ... % error term
                    obj.p_drv/2 * trace( J' * J ) + ... % derivative term
                    obj.p_reg/2 * trace( obj.W' * obj.W ) ;     % regularization term
            end
        end


        % error of model J = - W' * J_phi(X)
        function J = jacobian(obj, X)
            % X     states to compute Jacobian in as columns

            F = jac( X, obj.phi, obj.theta ) ;
            J = (- obj.W' * F) ;
        end

        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            f = ( obj.W' * feval(obj.phi, x) + obj.b ) ;
        end
    end
end