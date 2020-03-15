% @Author: OctaveOliviers
% @Date:   2020-02-22 15:30:10
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-14 18:04:21

% compute Jacobian matrix in each pattern as long matrix
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen feature map
%       varargin    : (1) parameters of feature map

function J = jacobian_matrix(patterns, type, varargin)


    [dim_patterns, num_patterns] = size(patterns);
    N = dim_patterns ;
    P = num_patterns ;

    type = lower(type);
    switch type
    
        case 'poly'
            deg = varargin{1} ;           
            J = zeros( N, N*P ) ;
            for p=1:P
                J(:, (p-1)*N+1:p*N) = diag( deg * patterns(:, p).^(deg-1)) ; 
            end
            
        case 'poly_2'
            assert(N == 1) ;
            deg = varargin{1} ;           
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

        case 'sign'
            J = zeros( N, N*P ) ;
            
    end
end