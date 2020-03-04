% return Kernel matrix
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen kernel function
%       param       : kernel parameter

function [Ftf, ftF, FtF] = kernel_derivatives(patterns, type, param)

    % extract useful information
    dim_patterns = size(patterns, 1) ;
    num_patterns = size(patterns, 2) ;
    
        
    % allocate memory
    Ftf = zeros( dim_patterns*num_patterns, num_patterns ) ;
    ftF = zeros( num_patterns,              dim_patterns*num_patterns ) ;
    FtF = zeros( dim_patterns*num_patterns, dim_patterns*num_patterns ) ;
    
    type = lower(type);
    switch type
            
        case { 'rbf', 'gauss', 'gaus' }
            sig = param ;
            K = kernel_matrix(patterns, type, param) ;

            for j = 1:num_patterns
                for i = 1:num_patterns
                    % shorten notation
                    x = patterns(:, i) ;
                    y = patterns(:, j) ;
                    
                    Ftf(1+(i-1)*dim_patterns:i*dim_patterns, j) = ...
                        K(i, j) * (y-x) / sig^2 ;
                    
                    ftF(i, 1+(j-1)*dim_patterns:j*dim_patterns) = ...
                        K(i, j) * (x-y)' / sig^2 ;
                    
                    FtF(1+(i-1)*dim_patterns:i*dim_patterns, 1+(j-1)*dim_patterns:j*dim_patterns) = ...
                        K(i, j) * (eye(dim_patterns)/sig^2 + (y-x)*(x-y)'/sig^4 ) ;
                end
            end
            
            
%         case 'softmax'
%             sig = param;
%             for j = 1:num_patterns
%                 matrix(j, j) = 1;
%                 for i = j+1:num_patterns
%                     diff = patterns(:, i) - patterns(:, j);
%                     k = exp( - (diff'*diff) / (2*sig^2) ) ;
%                     matrix(i, j) = k;
%                     matrix(j, i) = k;
%                 end
%             end
%             matrix = matrix ./ sum(matrix, 2);
    end
end