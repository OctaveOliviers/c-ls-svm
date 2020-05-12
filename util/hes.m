% Created  by OctaveOliviers
%          on 2020-03-29 16:54:38
%
% Modified on 2020-05-08 15:05:59

% compute Hessian of each component of the feature map in each pattern
%   input
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen feature map
%       varargin    : (1) parameters of feature map
%   output
%       H           : matrix fo size [ dim patterns , (dim patterns x num patterns), dim dual space ]
%
% only for explicitely computable feature maps

function H = hes( patterns, type, varargin )

    % extract useful parameters
    [N, P] = size(patterns);

    type = lower(type);
    switch type

        case 'tanh'
            H = zeros( N, N*P, N ) ;
            % indices of diagonal elements in (N x N x N) matrix
            diag_idx = 1:(P*N^2+N+1):((N-1)*P*N^2+(N-1)*N+N) ;
            for p=1:P
                H( (p-1)*N^2 + diag_idx ) = -2 * tanh( patterns(:, p) ) .* ( 1 - tanh( patterns(:, p) ).^2 ) ;
            end

        case 'sigmoid'
            H = zeros( N, N*P, N ) ;
            % indices of diagonal elements in (N x N x N) matrix
            diag_idx = 1:(P*N^2+N+1):((N-1)*P*N^2+(N-1)*N+N) ;
            for p=1:P
                H( (p-1)*N^2 + diag_idx ) = sigmoid(patterns(:, p)) .* ( 1 - sigmoid(patterns(:, p)) ) .* ( 1 - 2*sigmoid(patterns(:, p)) ) ;
            end

        case 'sign'
            H = zeros( N, N*P, N ) ;
            
    end
end