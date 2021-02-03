% Created  by OctaveOliviers
%          on 2020-09-25 17:21:52
%
% Modified on 2020-09-27 16:07:17

% softmax: normalized exponential function of columns of matrix
function F = softmax( X )
    
    F = exp(X) ./ sum(exp(X), 1) ;

end