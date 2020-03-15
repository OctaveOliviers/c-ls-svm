% @Author: OctaveOliviers
% @Date:   2020-03-04 22:56:29
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-14 18:02:14

% compute kernel matrix 
%       m = phi(x)^T * phi(y) 
% for data in X and Y

function m = phiTphi(X, Y, fun, varargin) 
    % X, Y      data matrix with observations in columns
    % fun 		feature map
    % varargin  (1) parameters of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    
    m = zeros(num_x, num_y) ;
    
    fun = lower(fun);
    switch fun

        case { 'rbf', 'gaussian', 'gauss', 'gaus', 'g' }
            sig = varargin{1} ;
            for j = 1:num_y
                for i = 1:num_x
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, j) = exp( -(x-y)'*(x-y) / (2*sig^2) ) ;
                end
            end 

        case { 'polynomial', 'poly', 'pol', 'p' }
            param   = varargin{1} ;
            assert( ndims(param)==2 , 'Polynomial kernel requires two parameters.' ) ;
            deg     = param(1) ;
            t       = param(2) ;
            for j = 1:num_y
                for i = 1:num_x
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, j) = ( x'*y + t )^deg ;
                end
            end

        case { 'tanh' }
            phi_x   = tanh(X) ;
            phi_y   = tanh(Y) ;
            m       = phi_x' * phi_y ;

        case { 'sign' }
            phi_x   = sign(X) ;
            phi_y   = sign(Y) ;
            m       = phi_x' * phi_y ; 

    end
end