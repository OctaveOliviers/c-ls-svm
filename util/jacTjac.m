% Created  by OctaveOliviers
%          on 2020-03-16 22:59:46
%
% Modified on 2020-12-09 21:25:55

% compute product of jacobians 
%       m = J_phi(x)^T * J_phi(y) 
% for data in X and Y

function m = jacTjac(X, Y, fun, varargin) 
    % X, Y      data matrix with observations in columns
    % fun       feature map
    % varargin  (1) parameters of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    dim_x = size(X, 1) ;
    dim_y = size(Y, 1) ;
    
    assert( dim_x==dim_y ) ;
    
    m = zeros( num_x*dim_x, num_y*dim_y) ;
    
    fun = lower(fun);
    switch fun
        
        case { 'rbf', 'gaussian', 'gauss', 'gaus', 'g' }
            sig = varargin{1} ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(1+(i-1)*dim_x:i*dim_x, 1+(j-1)*dim_y:j*dim_y) = ...
                        exp(-(x-y)'*(x-y)/(2*sig^2)) * ( eye(length(x))/sig^2 + (y-x)*(x-y)'/sig^4 ) ;
                end
            end

        case { 'polynomial', 'poly', 'pol', 'p' }
            param   = varargin{1} ;
            assert( ndims(param)==2 , 'Polynomial kernel requires two parameters.' ) ;
            deg     = param(1) ;
            t       = param(2) ;
            
            if (deg>1)
                for i = 1:num_x
                    for j = 1:num_y
                        x = X(:, i) ;
                        y = Y(:, j) ;
                        m(1+(i-1)*dim_x:i*dim_x, 1+(j-1)*dim_y:j*dim_y) = ...
                            deg * ( x'*y + t )^(deg-1) * eye(dim_x) + ...
                            deg * (deg-1) * ( x'*y + t )^(deg-2) * y*x' ;
                    end
                end
            else
                m = repmat( eye(dim_x), [num_x, 1] ) * repmat( eye(dim_y), [1, num_y] ) ;
            end

        case { 'tanh' }
            jac_x   = jac(X, 'tanh') ;
            jac_y   = jac(Y, 'tanh') ;
            m       = jac_x' * jac_y ;

        case { 'sign' }
            jac_x   = jac(X, 'sign') ;
            jac_y   = jac(Y, 'sign') ;
            m       = jac_x' * jac_y ;

    end
end