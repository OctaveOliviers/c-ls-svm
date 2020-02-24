% return Jacobian matrics in each pattern as long matrix
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen kernel function
%       param       : kernel parameter

function J = jacobian_matrix(patterns, type, param)

    [dim_patterns, num_patterns] = size(patterns);
    N = dim_patterns ;
    P = num_patterns ;

    type = lower(type);
    switch type
    
        case 'poly'
            deg = param ;           
            J = zeros( N, N*P ) ;
            for p=1:P
              J(:, (p-1)*N+1:p*N) = diag( deg * patterns(:, p).^(deg-1)) ; 
            end
            
        case 'poly_2'
            assert(N == 1) ;
            deg = param ;           
            J = zeros( deg, N*P ) ;
            for p=1:P
                for d=1:deg
                    J(d, (p-1)*N+1:p*N) = deg*patterns(:, p).^(deg-1) ;                     
                end
            end
            
        case 'tanh'
            J = zeros( N, N*P ) ;
            for p=1:P
              J(:, (p-1)*N+1:p*N) = diag( 1./ cosh( patterns(:, p) ).^2) ; 
            end
            
    end
end