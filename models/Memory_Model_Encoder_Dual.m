% Created  by OctaveOliviers
%          on 2020-03-29 19:05:54
%
% Modified on 2020-04-10 15:03:17

classdef Encoder_Dual
    
    properties
        % architecture
        phi
        theta
        % hyper-parameters
        p_drv
        p_reg
    end

    methods
        % constructor
        function obj = Encoder(phi, theta, p_err, p_drv, p_reg)
            % architecture
            obj.num_lay = 1 ;
            obj.phi     = phi ;     % string
            obj.theta   = theta ;   % float
            % hyper-parameters
            obj.p_drv   = p_drv ;   % float
            obj.p_reg   = p_reg ;   % float
            % model information
            obj.name    = join([ '1-layered encoder (', phi, ')']) ;
        end


        % train model for objective p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
        function obj = train(obj, X, varargin)
            % X         patterns to memorize
            % varargin  contains dimension of output space if it is different form input

            % extract useful parameters
            [N_i, P] = size(X) ;
            obj.X    = X ;

            if ( nargin<3 )
                N_o = N_i ;             % input and output dimensions match
            else
                N_o = varargin{1} ;     % input and output dimensions are different
            end 

            % build kernel term
            jTj = jacTjac(X, X, obj.phi, obj.theta) ;
                
            % matrices for linear system AX=B
            A = - jTj/obj.p_reg ;
            
            % compute parameters
            [eig_vec, eig_val] = eigs(A, N_o, 'smallestabs') 
            
            obj.L_d = eig_vec' ;

            % for computable feature map also compute W
            % if ( strcmp(obj.phi, 'tanh') | strcmp(obj.phi, 'sign') )
            %     F = jac( X, obj.phi ) ;
            %     obj.W = 1/obj.p_reg * ( F*obj.L_d' ) ;
            % end

            % store error, jacobian and lagrangian
            % obj.E = obj.error( X, Y ) ;
            obj.J = obj.L_d/obj.p_drv ;
            % obj.L = obj.lagrangian( ) ;

            % store latent representation of each pattern
            obj.Y = obj.simulate_one_step( X ) ; 

            disp("contractive model trained in dual")
        end


        % compute value of Lagrangian
        function L = lagrangian(obj, varargin)

        end


        % error of model on derivative J = - W' * J_phi(X)
        function J = jacobian(obj, X)
            % X     states to compute Jacobian in as columns

            % extract useful parameters
            [Nx, P] = size(X) ;
            [Ny, ~] = size(obj.Y) ;

            J = zeros( Ny, Nx*P ) ;
            for p = 1:size(X, 2)
                PTf = phiTjac( obj.X, X(:, p), obj.phi, obj.theta ) ;
                FTf = jacTjac( obj.X, X(:, p), obj.phi, obj.theta ) ;
                J(:, (p-1)*Nx+1:p*Nx) = - 1/obj.p_reg * ( obj.L_e*PTf + obj.L_d*FTf ) ;
            end
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            FTp = jacTphi( obj.X, x, obj.phi, obj.theta ) ;

            f   = -(obj.L_d*FTp)/obj.p_reg ; % + obj.b ;
        end
    end
end