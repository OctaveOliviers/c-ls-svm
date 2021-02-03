% Created  by OctaveOliviers
%          on 2020-05-08 11:29:17
%
% Modified on 2020-05-08 11:38:11

% element-wise computation of sigmoid

function y = sigmoid(x)
    
    y = 1./( 1 + exp(-x) ) ;

end