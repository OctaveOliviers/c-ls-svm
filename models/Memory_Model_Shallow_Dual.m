% @Author: OctaveOliviers
% @Date:   2020-03-15 16:25:15
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 22:12:35

classdef Memory_Model_Shallow_Dual < Memory_Model_Shallow
	
	methods
		% constructor
		function obj = Memory_Model_Shallow_Dual(phi, theta, p_err, p_drv, p_reg)
			% superclass constructor
			obj@Memory_Model_Shallow(phi, theta, p_err, p_drv, p_reg) ;
			% subclass secific variable
			obj.space = 'dual' ;
		end


		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(J^TJ) + p_reg/2*Tr(W^TW)
		function obj = train(obj, X, varargin)
			% X 		patterns to memorize
			% varargin	contains Y to map patterns X to (for stacked architectures)
			
			% extract useful parameters
			[N, P] 			= size(X) ;
			obj.patterns 	= X ;

			if ( nargin<3 )
				Y = X ;
			else
				Y = varargin{1} ;
			end

			% build kernel terms
			pTp = phiTphi(X, X, obj.phi, obj.theta) ;
			pTj = phiTjac(X, X, obj.phi, obj.theta) ;
			jTp = jacTphi(X, X, obj.phi, obj.theta) ;
			jTj = jacTjac(X, X, obj.phi, obj.theta) ;
			    
			% % matrices for linear system AX=B
			A = zeros( P+P*N+1, P+P*N+1 ) ;
			B = zeros( P+P*N+1, N ) ;

			% left-hand side
			A(1:P,       1:P)       = pTp/obj.p_reg + eye(P)/obj.p_err ;
			A(1:P,       P+1:end-1) = pTj/obj.p_reg ;
			A(P+1:end-1, 1:P)       = jTp/obj.p_reg ;
			A(P+1:end-1, P+1:end-1) = jTj/obj.p_reg + eye(P*N)/obj.p_drv ;
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

		    % disp("model trained in dual")
		end


		% compute value of Lagrangian
		function L = lagrangian(obj)
			% error term
			E = obj.model_error( obj.patterns ) ;
			% derivative term
			J = obj.model_jacobian( obj.patterns ) ;
			% regularization term
			PTP = phiTphi( obj.patterns, obj.patterns, obj.phi, obj.theta ) ;
			PTF = phiTjac( obj.patterns, obj.patterns, obj.phi, obj.theta ) ;
			FTP = jacTphi( obj.patterns, obj.patterns, obj.phi, obj.theta ) ;
			FTF = jacTjac( obj.patterns, obj.patterns, obj.phi, obj.theta ) ;
			WTW = 1/obj.p_reg * ( obj.L_e*PTP*obj.L_e' + obj.L_e*PTF*obj.L_d' ...
								+ obj.L_d*FTP*obj.L_e' + obj.L_d*FTF*obj.L_d' ) ;

			L = obj.p_err/2 * trace(E'*E) + obj.p_drv/2 * trace(J'*J) + obj.p_reg/2 * trace(WTW) ;
		end


		% error of model E = X - W' * phi(X) - B
		function E = model_error(obj, X)
			% X		states to compute error in

			E = X - obj.simulate_one_step( X ) ;
		end


		% error of model J = - W' * J_phi(X)
		function J = model_jacobian(obj, X)
			% X 	states to compute Jacobian in as columns

			% extract useful parameters
			[N, P] = size(X) ;

			J = zeros( N, N*P ) ;
			for p = 1:size(X, 2)
				PTf = phiTjac( obj.patterns, X(:, p), obj.phi, obj.theta ) ;
				FTf = jacTjac( obj.patterns, X(:, p), obj.phi, obj.theta ) ;
				J(:, (p-1)*N+1:p*N) = - 1/obj.p_reg * ( obj.L_e*PTf + obj.L_d*FTf ) ;
			end
		end


		% simulate model over one step
		function f = simulate_one_step(obj, x)
			% x		matrix with start positions to simulate from as columns

			PTp = phiTphi( obj.patterns, x, obj.phi, obj.theta ) ;
			FTp = jacTphi( obj.patterns, x, obj.phi, obj.theta ) ;

			f   = (obj.L_e*PTp + obj.L_d*FTp)/obj.p_reg + obj.b ;
		end
	end
end