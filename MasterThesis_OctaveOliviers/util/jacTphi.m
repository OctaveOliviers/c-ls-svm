% Created  by OctaveOliviers
%          on 2020-03-16 22:59:46
%
% Modified on 2020-04-27 22:21:58

% compute product of jacobian with feature map 
%       m = J_phi(x)^T * phi(y) 
% for data in X and Y

function m = jacTphi(X, Y, fun, varargin) 
    % X, Y      data matrix with observations in columns
    % fun       feature map
    % varargin  (1) parameters of feature map


    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    dim_x = size(X, 1) ;
    dim_y = size(Y, 1) ;
    
    assert( dim_x==dim_y ) ;
    
    m = zeros( num_x*dim_x, num_y) ;
    
    fun = lower(fun);
    switch fun
        
        case { 'rbf', 'gaussian', 'gauss', 'gaus', 'g' }
            sig = varargin{1} ;
            for i = 1:num_x
                for j = 1:num_y
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(1+(i-1)*dim_x:i*dim_x, j) = ...
                        exp( -(x-y)'*(x-y) / (2*sig^2) ) * (y-x) / sig^2 ;
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
                    m(1+(i-1)*dim_x:i*dim_x, j) = ...
                        deg * ( x'*y + t )^(deg-1) * y ;
                end
            end

        case { 'tanh' }
            jac_x   = jac(X, 'tanh') ;
            phi_y   = tanh(Y) ;
            m       = jac_x' * phi_y ;

        case { 'sign' }
            jac_x   = jac(X, 'sign') ;
            phi_y   = sign(Y) ;
            m       = jac_x' * phi_y ;

    end
end