% Created  by OctaveOliviers
%          on 2020-09-25 17:21:52
%
% Modified on 2020-09-26 15:36:46

% softmax: normalized exponential function
function f = softmax( x )
    
    f = exp(x) ./ sum(exp(x)) ;

end