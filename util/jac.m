% Created  by OctaveOliviers
%          on 2020-02-22 15:30:10
%
% Modified on 2020-04-11 22:10:37

% compute Jacobian matrix in each pattern as long matrix
%   input
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen feature map
%       varargin    : (1) parameters of feature map
%   output
%       J           : matrix of size [ dim patterns , (dim patterns x num patterns) ]
%
% only for explicitely computable feature maps

function J = jac( patterns, type, varargin )

    % extract useful parameters
    [N, P] = size(patterns);

    type = lower(type);
    switch type

        case 'tanh'
            J = zeros( N, N*P ) ;
            for p=1:P
                J(:, (p-1)*N+1:p*N) = diag( 1./ cosh( patterns(:, p) ).^2) ; 
            end

        case 'sign'
            J = zeros( N, N*P ) ;
            
    end
end