% Created  by OctaveOliviers
%          on 2020-03-15 16:25:40
%
% Modified on 2020-04-11 22:08:58

classdef Layer_Dual < Layer
    
    methods

        % constructor
        function obj = Layer_Dual(phi, theta, p_err, p_drv, p_reg)
            % superclass constructor
            obj@Layer(phi, theta, p_err, p_drv, p_reg) ;
            % subclass secific variable
            obj.space = 'dual' ;
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
                assert( size(X, 2)==size(Y, 2),  'Number of patterns in X and Y do not match.' ) ;
            end 

            % extract useful parameters
            [Nx, P] = size(X) ;
            [Ny, ~] = size(Y) ;
            obj.X   = X ;
            obj.Y   = Y ;

            % build kernel terms
            pTp = phiTphi(X, X, obj.phi, obj.theta) ;
            pTj = phiTjac(X, X, obj.phi, obj.theta) ;
            jTp = jacTphi(X, X, obj.phi, obj.theta) ;
            jTj = jacTjac(X, X, obj.phi, obj.theta) ;
                
            % matrices for linear system AX=B
            A = zeros( P+P*Nx+1, P+P*Nx+1 ) ;
            B = zeros( P+P*Nx+1, Ny ) ;

            % left-hand side
            A(1:P,       1:P)       = pTp/obj.p_reg + eye(P)/obj.p_err ;
            A(1:P,       P+1:end-1) = pTj/obj.p_reg ;
            A(P+1:end-1, 1:P)       = jTp/obj.p_reg ;
            A(P+1:end-1, P+1:end-1) = jTj/obj.p_reg + eye(P*Nx)/obj.p_drv ;
            A(1:P,       end)       = 1 ;
            A(end,       1:P)       = 1 ;
                
            % right-hand side
            B(1:P, :) = Y' ;

            % compute parameters
            v = A\B ;
            % primal
            obj.b   = v(end, :)' ;
            % dual
            obj.L_e = v(1:P, :)' ;
            obj.L_d = v(P+1:end-1, :)' ;

            % for computable feature map also compute W
            if ( strcmp(obj.phi, 'tanh') | strcmp(obj.phi, 'sign') )
                Phi = feval(obj.phi, X) ;
                Jac = jac( X, obj.phi ) ;
                obj.W = 1/obj.p_reg * ( Phi*obj.L_e' + Jac*obj.L_d' ) ;
            elseif ( strcmp(obj.phi, 'poly') | obj.theta==[1, 0] )
                Phi = X ;
                Jac = repmat( eye(Nx), [1, P] ) ;
                obj.W = 1/obj.p_reg * ( Phi*obj.L_e' + Jac*obj.L_d' ) ;
            end

            % store layer error, jacobian and Lagrange function
            obj = obj.store_lagrange_param() ;

            % disp("model trained in dual")
        end


        % error of model E = X - W' * phi(X) - B
        function E = layer_error(obj, varargin)
            % X     states to compute error in

            % compute error of model
            if ( nargin < 2 )
                E = obj.E ;

            % compute error in new point
            else
                E = varargin{1} - obj.simulate_one_step( varargin{1} ) ;
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
                % extract useful parameters
                X = varargin{1} ;
                [N_in,  P] = size(X) ;
                [N_out, ~] = size(obj.Y) ;

                assert( size(obj.X, 1) == N_in, ...
                        "The dimension of the data in X does not match the dimension of the training data.") ;

                J = zeros( N_out, N_in*P ) ;
                for p = 1:P
                    PTf = phiTjac( obj.X, X(:, p), obj.phi, obj.theta ) ;
                    FTf = jacTjac( obj.X, X(:, p), obj.phi, obj.theta ) ;
                    J(:, (p-1)*N_in+1:p*N_in) = - 1/obj.p_reg * ( obj.L_e*PTf + obj.L_d*FTf ) ;
                end
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
                % compute norm of weights
                PTP = phiTphi( obj.X, obj.X, obj.phi, obj.theta ) ;
                PTF = phiTjac( obj.X, obj.X, obj.phi, obj.theta ) ;
                FTP = jacTphi( obj.X, obj.X, obj.phi, obj.theta ) ;
                FTF = jacTjac( obj.X, obj.X, obj.phi, obj.theta ) ;
                WTW = 1/obj.p_reg^2 * ( obj.L_e*PTP*obj.L_e' + obj.L_e*PTF*obj.L_d' ...
                                      + obj.L_d*FTP*obj.L_e' + obj.L_d*FTF*obj.L_d' ) ;

                L = obj.p_err/2 * trace( E' * E ) + ... % error term
                    obj.p_drv/2 * trace( J' * J ) + ... % derivative term
                    obj.p_reg/2 * trace( WTW ) ;        % regularization term
            end
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            PTp = phiTphi( obj.X, x, obj.phi, obj.theta ) ;
            FTp = jacTphi( obj.X, x, obj.phi, obj.theta ) ;

            f   = (obj.L_e*PTp + obj.L_d*FTp)/obj.p_reg + obj.b ;
        end
    end
end