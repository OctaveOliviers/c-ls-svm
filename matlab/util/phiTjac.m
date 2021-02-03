% Created  by OctaveOliviers
%          on 2020-03-16 22:59:46
%
% Modified on 2020-04-27 22:20:30

% compute product of feature map with jacobian 
%       m = phi(x)^T * J_phi(y) 
% for data in X and Y

function m = phiTjac(X, Y, fun, varargin) 
    % X, Y      data matrix with observations in columns
    % fun       feature map
    % varargin  (1) parameters of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    dim_x = size(X, 1) ;
    dim_y = size(Y, 1) ;
    
    assert( dim_x==dim_y ) ;
    
    m = zeros( num_x, num_y*dim_y) ;
    
    fun = lower(fun);
    switch fun
    
        case { 'rbf', 'gaussian', 'gauss', 'gaus', 'g' }
            sig = varargin{1} ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, 1+(j-1)*dim_y:j*dim_y) = ...
                        exp( -(x-y)'*(x-y) / (2*sig^2) ) * (x-y)' / sig^2 ;
                end
            end

        case { 'polynomial', 'poly', 'pol', 'p' }
            param   = varargin{1} ;
            assert( ndims(param)==2 , 'Polynomial kernel requires two parameters.' ) ;
            deg     = param(1) ;
            t       = param(2) ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, 1+(j-1)*dim_y:j*dim_y) = ...
                        deg * ( x'*y + t )^(deg-1) * x' ;
                end
            end

        case { 'tanh' }
            phi_x   = tanh(X) ;
            jac_y   = jac(Y, 'tanh') ;            
            m       = phi_x' * jac_y ;

        case { 'sign' }
            phi_x   = sign(X) ;
            jac_y   = jac(Y, 'tanh') ;            
            m       = phi_x' * jac_y ;
            
    end
end