% @Author: OctaveOliviers
% @Date:   2020-03-15 16:25:15
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-29 12:30:29

classdef Memory_Model_Shallow_Dual < Memory_Model_Shallow
	
	methods
		% constructor
		function obj = Memory_Model_Shallow_Dual(phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj@Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg) ;
			% subclass secific variable
			obj.space = 'dual' ;
			% model information
			obj.name 	= join([ '1-layered network (', phi, ')']) ;
		end


		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
		function obj = train(obj, X, varargin)
			% X 		patterns to memorize
			% varargin	contains Y to map patterns X to (for stacked architectures)
			
			if ( nargin<3 )
				Y = X ;
			else
				Y = varargin{1} ;

				% check correctness of input
				assert( size(X, 2)==size(Y, 2),  'Number of patterns in X and Y do not match.' ) ;
			end	

			% extract useful parameters
			[Nx, P]	= size(X) ;
			[Ny, ~]	= size(Y) ;
			obj.X 	= X ;
			obj.Y 	= Y ;

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
			A(1:P, 		 end)       = 1 ;
			A(end,       1:P)       = 1 ;
			    
			% right-hand side
			B(1:P, :) = Y' ;

			% compute parameters
			v = A\B ;
			% primal
			obj.b 	= v(end, :)' ;
			% dual
			obj.L_e	= v(1:P, :)' ;
			obj.L_d	= v(P+1:end-1, :)' ;

			% for computable feature map also compute W
			if ( strcmp(obj.phi, 'tanh') | strcmp(obj.phi, 'sign') )
				P = feval(obj.phi, X) ;
				F = jac( X, obj.phi ) ;
				obj.W = 1/obj.p_reg * ( P*obj.L_e' + F*obj.L_d' ) ;
			end

			% store error and jacobian
			obj.E = obj.error( X, Y ) ;
			obj.J = obj.jacobian( X ) ;

		    disp("model trained in dual")
		end


		% compute value of Lagrangian
		function L = lagrangian(obj, varargin)

			% compute norm of weights
			PTP = phiTphi( obj.X, obj.X, obj.phi, obj.theta ) ;
			PTF = phiTjac( obj.X, obj.X, obj.phi, obj.theta ) ;
			FTP = jacTphi( obj.X, obj.X, obj.phi, obj.theta ) ;
			FTF = jacTjac( obj.X, obj.X, obj.phi, obj.theta ) ;
			WTW = 1/obj.p_reg * ( obj.L_e*PTP*obj.L_e' + obj.L_e*PTF*obj.L_d' ...
								+ obj.L_d*FTP*obj.L_e' + obj.L_d*FTF*obj.L_d' ) ;

			% compute lagrangian of model
			if ( nargin < 2 )
				L = obj.p_err/2 * trace( obj.E' * obj.E ) + ...	% error term
					obj.p_drv/2 * trace( obj.J' * obj.J ) + ...	% derivative term
					obj.p_reg/2 * trace( WTW ) ;				% regularization term

			% evaluate lagrangian with new parameters
			else
				X = varargin{1} ;
				Y = varargin{2} ;
				
				E = Y - obj.simulate_one_step( X ) ; 
				J = obj.jacobian( X ) ;

				L = obj.p_err/2 * trace( E' * E ) + ...	% error term
					obj.p_drv/2 * trace( J' * J ) + ...	% derivative term
					obj.p_reg/2 * trace( WTW ) ;		% regularization term
			end
		end


		% error of model on derivative J = - W' * J_phi(X)
		function J = jacobian(obj, X)
			% X 	states to compute Jacobian in as columns

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
			% x		matrix with start positions to simulate from as columns

			PTp = phiTphi( obj.X, x, obj.phi, obj.theta ) ;
			FTp = jacTphi( obj.X, x, obj.phi, obj.theta ) ;

			f   = (obj.L_e*PTp + obj.L_d*FTp)/obj.p_reg + obj.b ;
		end
	end
end