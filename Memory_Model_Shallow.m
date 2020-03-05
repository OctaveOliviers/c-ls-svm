% @Author: OctaveOliviers
% @Date:   2020-03-05 09:51:23
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 15:47:44

classdef Memory_Model_Shallow < Memory_Model
	
	properties

	end

	methods
		% constructor
		function obj = Memory_Model_Shallow(space, phi, theta, p_err, p_drv, p_reg)
			% architecture
			obj.space 	= space ;
			obj.phi 	= phi ;
			obj.theta 	= theta ;
			% hyper-parameters
			obj.p_err 	= p_err ;
			obj.p_drv 	= p_drv ;
			obj.p_reg 	= p_reg ;
		end


		% train model for objective p_err/2*Tr(E^TE) + p_drv/2*Tr(JJ^T) + p_reg/2*Tr(W^TW)
		function obj = train(obj, X, varargin)
			% X 		patterns to memorize
			% varargin	contains Y to map patterns X to (for stacked architectures)
			
			% extract useful parameters
			[N, P] 			= size(X) ;
			obj.patterns 	= X ;

			if ( size(varargin)==0 )
				Y = X ;
			else
				Y = varargin{1} ;
			end

			switch obj.space

				case {'primal', 'p'}
					% feature map in each data point
					f = feval(obj.phi, X) ;
					% jacobians of feature map in each data points
					F = jacobian_matrix(X, obj.phi, obj.theta) ;
					% dimension of dual space
					D = size(f, 1) ;

					% matrices for linear system AX=B
					A = zeros( D+1, D+1 ) ;
					B = zeros( D+1, N ) ;

					% left-hand side
					A( 1:D, 1:D ) = f*f' + obj.p_reg*F*F'/obj.p_err ;
					A( 1:D, end ) = sum(f, 2) ;
					A( end, 1:D ) = sum(f, 2) ;
					A( end, end ) = P ;

					% right-hand side
					B( 1:D, : ) = f*Y' ;
					B( end, : ) = sum(Y, 2) ;

					% compute parameters
					v = A\B ;
					%
					obj.W = v(1:N, :) ;
					obj.b = v(end, :)' ;
		            
		        case {'dual', 'd'}
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
					%
					obj.L_e	= v(1:P, :)' ;
					obj.L_d	= v(P+1:end-1, :)' ;
					obj.b 	= v(end, :)' ;
		    end
		    disp("model trained")
		end


		% simulate model over one step
		function f = simulate_one_step(obj, x)
			% x		matrix with start positions to simulate from as columns

			switch obj.space

				case {"primal", "p"}
		            f = obj.W' * feval(obj.phi, x) + obj.b ;

		        case {"dual", "d"}
					pTp = phiTphi(obj.patterns, x, obj.phi, obj.theta) ;
					jTp = jacTphi(obj.patterns, x, obj.phi, obj.theta) ;

					f   = (obj.L_e*pTp + obj.L_d*jTp)/obj.p_reg + obj.b ;
		    end
		end
	end
end