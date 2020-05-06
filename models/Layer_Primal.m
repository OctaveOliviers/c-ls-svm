% Created  by OctaveOliviers
%          on 2020-03-15 16:25:40
%
% Modified on 2020-05-05 15:51:04

classdef Layer_Primal < Layer
    
    methods

        % constructor
        function obj = Layer_Primal(varargin)
            % superclass constructor
            obj@Layer(varargin{:}) ;
            % subclass secific variable
            obj.space = 'primal' ;
        end


         % train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
        function obj = train(obj, X, varargin)
            % X         patterns to memorize
            % varargin  contains Y to map patterns X to (for stacked architectures)
            
            if ( nargin<3 )
                Y = X ;

                % check correctness of input
                assert( size(X, 1)==obj.N_out,   'Number of neurons in layer and X do not match.' ) ;
            else
                Y = varargin{1} ;

                % check correctness of input
                assert( size(Y, 1)==obj.N_out,   'Number of neurons in layer and Y do not match.' ) ;
                assert( size(Y, 2)==size(X, 2),  'Number of patterns in X and Y do not match.' ) ;
            end

            % extract useful parameters
            [Nx, P]   = size(X) ;
            [Ny, ~]   = size(Y) ;
            obj.X     = X ;
            obj.Y     = Y ;
            obj.N_in  = Nx ;
            obj.P     = P ;

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
            obj = obj.store_lagrange_param() ;

            % disp("model trained in primal")
        end


        % error of model E = X - W' * phi(X) - B
        function E = layer_error(obj, varargin)
            % X     states to compute error in

            % compute error of model
            if (nargin<2)
                E = obj.E ;

            % compute error in new point
            elseif ( nargin == 3 )
                E = varargin{2} - obj.simulate_one_step( varargin{1} ) ;
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
                J = (obj.W' * F) ;
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

                L = obj.p_err/2 * trace( E' * E ) + ...         % error term
                    obj.p_drv/2 * trace( J' * J ) + ...         % derivative term
                    obj.p_reg/2 * trace( obj.W' * obj.W ) ;     % regularization term
            end
        end


        % compute gradient of Lagrangian with respect to its input evaluated in columns of X
        function grad = gradient_lagrangian_wrt_input(obj)

            % gradient of error
            dE = zeros(obj.N_in, obj.P) ;
            for p = 1:obj.P
                dE(:, p) = obj.J(:, 1+(p-1)*obj.N_in:p*obj.N_in)' * obj.E(:, p) ;
            end

            % gradient of jacobian
            dJ  = zeros(obj.N_in, obj.P) ;
            % 3D hessian
            Hes = hes( obj.X, obj.phi, obj.theta ) ;
            for p = 1:obj.P
                for n = 1:obj.N_in
                    dJ(n, p) = trace( squeeze(Hes(:, n+(p-1)*obj.N_in, :)) * obj.W * obj.J(:, 1+(p-1)*obj.N_in:p*obj.N_in) ) ;
                end
            end

            grad = obj.p_err * dE + obj.p_drv * dJ ;
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            f = ( obj.W' * feval(obj.phi, x) + obj.b ) ;
        end
    end
end