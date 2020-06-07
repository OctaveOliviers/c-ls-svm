% Created  by OctaveOliviers
%          on 2020-04-26 18:18:37
%
% Modified on 2020-04-27 22:07:04

% Simulate the Henon map
% https://mathworld.wolfram.com/HenonMap.html

% number of points to simulate
num_sim = 3 ;
% number of steps to simulate over
len_sim = 10 ;
% store path in the state space
path = zeros(2, len_sim, num_sim) ;

% parameters
a = 1.4 ;
b = 0.3 ;
% Henon map
map = @(x) [ 1 - a*x(1, :).^2 + x(2, :) ; b*x(1, :) ] ;

% points to start simulation
scale = 2 ;
prec  = 2*scale / (num_sim-1) ;
path(:, 1, :) = [ -scale:prec:scale ; -scale:prec:scale ] ;

% simulate
for i = 1:len_sim-1
    path(:, i+1, :) = map( path(:, i, :) ) ;
end

% visualize path
figure('position', [100, 100, 400, 500])
box on ;
hold on ;
for n = 1:num_sim
    plot( squeeze(path(1, :, n)), squeeze(path(2, :, n)), 'b-' )
end
hold off ;