% @Author: OctaveOliviers
% @Date:   2020-03-15 14:34:35
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 14:50:53

% compute Hessian of each component of the feature map in each pattern
%	input
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen feature map
%       varargin    : (1) parameters of feature map
%	output
%		H 			: matrix fo size [ dim patterns , (dim patterns x num patterns), dim dual space ]
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
                J(:, (p-1)*N+1:p*N) = diag( 1./ cosh( patterns(:, p) ).^2) ; 
            end

        case 'sign'
            H = zeros( N, N*P, N ) ;
            
    end
end