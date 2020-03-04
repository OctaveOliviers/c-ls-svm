classdef Memory_Model
	properties
		patterns
		% model architecture
		space 	% 'primal' or 'dual'
		phi		% feature map as string
		theta	% parameter of feature map
		% model hyper-parameters
		p_err	% importance of minimizing error
		p_drv	% importance of minimizing derivative
		p_reg	% importance of regularization
		% model parameters
		W 		% primal weights
		L 		% dual Lagrange parameters for error
		M 		% dual Lagrange parameters for derivative
		b		% bias
	end

	methods
		% constructor
		function obj = Memory_Model(space, phi, theta, p_err, p_drv, p_reg)
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
		function obj = train(obj, data)
			% data 		patterns to memorize

			% extract useful parameters
			[N, P] = size(data) ;
			obj.patterns = data ;

			switch obj.space

				case {'primal', 'p'}
					% feature map in each data point
					f = feval(obj.phi, data) ;
					% jacobians of feature map in each data points
					F = jacobian_matrix(data, obj.phi, obj.theta) ;
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
					B( 1:D, : ) = f*data' ;
					B( end, : ) = sum(data, 2) ;

					% compute parameters
					X = A\B ;
					%
					obj.W = X(1:N, :) ;
					obj.b = X(end, :)' ;
		            
		        case {'dual', 'd'}
					% build kernel terms
					ftf = phiTphi(data, data, obj.phi, obj.theta) ;
					ftF = phiTjac(data, data, obj.phi, obj.theta) ;
					Ftf = jacTphi(data, data, obj.phi, obj.theta) ;
					FtF = jacTjac(data, data, obj.phi, obj.theta) ;
					    
					% % matrices for linear system AX=B
					A = zeros( P+P*N+1, P+P*N+1 ) ;
					B = zeros( P+P*N+1, N ) ;

					% left-hand side
					A(1:P,       1:P)       = ftf/obj.p_reg + eye(P)/obj.p_err ;
					A(1:P,       P+1:end-1) = ftF/obj.p_reg ;
					A(P+1:end-1, 1:P)       = Ftf/obj.p_reg ;
					A(P+1:end-1, P+1:end-1) = FtF/obj.p_reg + eye(P*N)/obj.p_drv ;
					A(1:P, 		 end)       = 1 ;
					A(end,       1:P)       = 1 ;
					    
					% right-hand side
					B(1:P, :) = data' ;

					% compute parameters
					X = A\B ;
					%
					obj.L = X(1:P, :)' ;
					obj.M = X(P+1:end-1, :)' ;
					obj.b = X(end, :)' ;
		    end
		    disp("model trained")
		end

		% visualize model
		function visualize

		end

		% simulate model 
		function f = simulate(obj, x)
			% x		matrix with start positions to simulate from as columns

			switch obj.space

				case {'primal', 'p'}
		            f = obj.W' * feval(obj.phi, x) + obj.b ;

		        case {'dual', 'd'}
					ftf = phiTphi(obj.patterns, x, obj.phi, obj.theta) ;
					Ftf = jacTphi(obj.patterns, x, obj.phi, obj.theta) ;

					f = (obj.L*ftf + obj.M*Ftf)/obj.p_reg + obj.b ;
		    end
		end
	end
end