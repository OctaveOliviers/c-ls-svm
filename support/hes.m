% Created  by OctaveOliviers
%          on 2020-03-29 16:54:38
%
% Modified on 2020-03-30 16:36:23

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
            for p=1:P
                for d = 1:N
                    H(d, d+(p-1)*N, d) = -2 * tanh( patterns(d, p) ) * ( 1 - tanh( patterns(d, p) )^2 ); 
                end
            end

        case 'sign'
            H = zeros( N, N*P, N ) ;
            
    end
end