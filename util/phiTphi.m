% Created  by OctaveOliviers
%          on 2020-03-04 22:56:29
%
% Modified on 2020-04-16 18:29:02

% compute kernel matrix (similarity measure) 
%       m = phi(x)^T * phi(y) 
% for data in X and Y

function m = phiTphi(X, Y, fun, varargin) 
    % X, Y      data matrix with observations in columns
    % fun       feature map
    % varargin  (1) parameters of feature map

    % extract useful variables
    num_x = size(X, 2) ;
    num_y = size(Y, 2) ;
    
    m = zeros(num_x, num_y) ;
    
    fun = lower(fun);
    switch fun

        case { 'rbf', 'gaussian', 'gauss', 'gaus', 'g' }
            XtX = sum(X.^2,1)' * ones(1,num_y) ;
            YtY = sum(Y.^2,1)' * ones(1,num_x) ;
            m   = XtX + YtY' - 2*X'*Y ;
            m   = exp(-m./(2*varargin{1}^2)) ;

        case { 'polynomial', 'poly', 'pol', 'p' }
            param   = varargin{1} ;
            m = ( X'*Y + param(2) ) .^param(1) ;

        case { 'linear', 'lin' }
            m = X'*Y ;

        case { 'tanh' }
            phi_x   = tanh(X) ;
            phi_y   = tanh(Y) ;
            m       = phi_x' * phi_y ;

        case { 'sign' }
            phi_x   = sign(X) ;
            phi_y   = sign(Y) ;
            m       = phi_x' * phi_y ;

        case { 'L2', 'l2', 'euclidean' }
            XtX = sum(X.^2,1)' * ones(1,num_y) ;
            YtY = sum(Y.^2,1)' * ones(1,num_x) ;
            m   = XtX + YtY' - 2*X'*Y ;

        case { 'L1', 'l1', 'manhattan' }
            for j = 1:num_y
                for i = 1:num_x
                    x = X(:, i) ;
                    y = Y(:, j) ;
                    m(i, j) = sum(abs(x-y)) ;
                end
            end

    end
end