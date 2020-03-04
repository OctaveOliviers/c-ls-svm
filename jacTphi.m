% Octave Oliviers - March 4th 2020
%
% compute product of jacobian with feature map 
%       m = J_phi(x)^T * phi(y) 
% for data in X and Y

function m = Ftf(X, Y, fun, param) 
    % X, Y      data matrix with observations in columns
    % fun 		feature map
    % param 	parameter of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    dim_x = size(X, 1) ;
    dim_y = size(Y, 1) ;
    
    assert( dim_x==dim_y ) ;
    
    m = zeros( num_x*dim_x, num_y) ;
    
    fun = lower(fun);
    switch fun
        case { 'rbf', 'gauss', 'gaus' }
            sig = param ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(1+(i-1)*dim_x:i*dim_x, j) = ...
                        exp(-(x-y)'*(x-y)/(2*sig^2)) * (y-x) / sig^2 ;
                end
            end
    end
end