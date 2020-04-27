% Created  by OctaveOliviers
%          on 2020-03-15 16:25:40
%
% Modified on 2020-04-11 19:30:06

classdef Layer_Primal < Layer
    
    methods

        % constructor
        function obj = Layer_Primal(phi, theta, p_err, p_drv, p_reg)
            % superclass constructor
            obj@Layer(phi, theta, p_err, p_drv, p_reg) ;
            % subclass secific variable
            obj.space = 'primal' ;
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

            % store layer error, jacobian and Lagrange function
            obj.store_lagrange_param() ;

            % disp("model trained in primal")
        end


        % error of model E = X - W' * phi(X) - B
        function E = layer_error(obj, varargin)
            % X     states to compute error in

            % compute error of model
            if (nargin<2)
                E = obj.E ;

            % compute error in new point
            else
                E = X - obj.simulate_one_step( X ) ;
            end
        end


        % error of model J = - W' * J_phi(X)
        function J = layer_jacobian(obj, varargin)
            % X     states to compute Jacobian in as columns

            % compute jacobian of model
            if ( nargin < 2 )
                J = obj.J ;

            % compute jacobian in new point
            else
                X = varargin{1} ;
                
                F = jac( X, obj.phi, obj.theta ) ;
                J = (- obj.W' * F) ;
            end

            
        end


        % compute value of Lagrange function
        function L = layer_lagrangian(obj, varargin)
            
            % compute lagrangian of model
            if ( nargin < 2 )
                L = obj.L ;

            % evaluate lagrangian with new parameters
            else
                X = varargin{1} ;
                Y = varargin{2} ;
                
                E = obj.layer_error( X, Y ) ; 
                J = obj.layer_jacobian( X ) ;

                L = obj.p_err/2 * trace( E' * E ) + ... % error term
                    obj.p_drv/2 * trace( J' * J ) + ... % derivative term
                    obj.p_reg/2 * trace( obj.W' * obj.W ) ;     % regularization term
            end
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            f = ( obj.W' * feval(obj.phi, x) + obj.b ) ;
        end
    end
end