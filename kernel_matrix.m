% return Kernel matrix
%       patterns    : matrix of size num_neurons x num_patterns
%       type        : string that identifies the chosen kernel function
%       param       : kernel parameter

function [matrix, phi] = kernel_matrix(patterns, type, param)

    num_patterns = size(patterns, 2);
    matrix = zeros( num_patterns, num_patterns );

    type = lower(type);
    switch type
    
        case 'poly'
            deg = param ;
            phi = patterns.^deg ;
            matrix = phi'*phi ;
            
            %matrix = patterns' * patterns;
            %matrix = matrix.^deg;
            
        case 'poly_2'
            assert(size(patterns, 1) == 1)
            deg = param ;
            phi = zeros(deg, num_patterns) ;
            for d=1:deg
                phi(d, :) = patterns.^d ;
            end
            matrix = phi'*phi ;
                    
        case 'linear'
            matrix = patterns' * patterns;
        
        case 'rbf'
            sig = param;
            for j = 1:num_patterns
                matrix(j, j) = 1;
                for i = j+1:num_patterns
                    diff = patterns(:, i) - patterns(:, j);
                    k = exp( - (diff'*diff) / (2*sig^2) ) ;
                    matrix(i, j) = k;
                    matrix(j, i) = k;
                end
            end 
            
        case 'tanh'
            phi = tanh(patterns) ;
            matrix = phi'*phi ;
            
        case 'softmax'
            sig = param;
            for j = 1:num_patterns
                matrix(j, j) = 1;
                for i = j+1:num_patterns
                    diff = patterns(:, i) - patterns(:, j);
                    k = exp( - (diff'*diff) / (2*sig^2) ) ;
                    matrix(i, j) = k;
                    matrix(j, i) = k;
                end
            end
            matrix = matrix ./ sum(matrix, 2);
    end
end