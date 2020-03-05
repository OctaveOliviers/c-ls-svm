% @Author: OctaveOliviers
% @Date:   2020-03-04 22:56:40
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-05 15:26:25

% compute product of feature map with jacobian 
%       m = phi(x)^T * J_phi(y) 
% for data in X and Y

function m = phiTjac(X, Y, fun, param) 
    % X, Y      data matrix with observations in columns
    % fun 		feature map
    % param 	parameter of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    dim_x = size(X, 1) ;
    dim_y = size(Y, 1) ;
    
    assert( dim_x==dim_y ) ;
    
    m = zeros( num_x, num_y*dim_y) ;
    
    fun = lower(fun);
    switch fun
        case { 'rbf', 'gauss', 'gaus' }
            sig = param ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, 1+(j-1)*dim_y:j*dim_y) = ...
                        exp(-(x-y)'*(x-y)/(2*sig^2)) * (x-y)' / sig^2 ;
                end
            end
    end
end