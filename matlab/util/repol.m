% Created  by OctaveOliviers
%          on 2020-09-24 15:39:53
%
% Modified on 2020-09-26 15:36:49

% repol: rectified polynomial function
function f = repol( deg, x )
    
    f = ( max(0, x) ).^deg;

end