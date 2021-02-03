% Created  by OctaveOliviers
%          on 2020-04-26 18:36:03
%
% Modified on 2020-04-27 14:47:58

% Simulate the logistic map
% http://physics.ucsc.edu/~peter/242/logistic.pdf

clear all
clc

% number of points to simulate
num_sim = 1 ;
% number of steps to simulate over
len_sim = 10 ;
% store path in the state space
path = zeros(num_sim, 2*len_sim+1) ;

% parameter
r = 2.2 ;
% Logistic map
map = @(x) r * x .* (1-x) ;

% points to start simulation
prec  = 0.8 / (num_sim-1) ;
% path(:, 1) = 0.1:prec:0.9 ;
path(:, 1) = 0.1 ;

% simulate
for i = 1:len_sim
    path(:, 2*i)   = path(:, 2*i-1) ;
    path(:, 2*i+1) = map( path(:, 2*i) ) ;
end

orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;
% visualize path
figure('position', [100, 100, 300, 285])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
grid on ;
box on ;
hold on ;
line_log = plot( 0:0.01:1, map(0:0.01:1) , 'color', KUL_blue, 'linewidth', 1 ) ;
for n = 1:num_sim
    line_path   = plot( squeeze( path(n, 1:end-1) ), squeeze( path(n, 2:end) ), 'color', orange, 'linewidth', 1 ) ;
    point_start = plot( path(n, 1), path(n, 1), 'o', 'color', orange, 'linewidth', 1.5 ) ;
end
line_id = line([0 1], [0 1], 'color', [0, 0, 0]) ;
plot(0,0,'.', 'color',red, 'markersize', 25)
plot(0.55,0.55,'.', 'color',green, 'markersize', 25)
hold off ;
axis equal
xticks([0 0.55 1])
yticks([0 1])
set(gca,'FontSize',12)
xlabel('$x^{(k)}$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$x^{(k+1)}$', 'interpreter', 'latex', 'fontsize', 14)
legend( [line_log, point_start, line_path], {"$r \; x^{(k)} \; (1-x^{(k)})$", "Starting point", "Movement"} , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12)
title( strcat('\textbf{Logistic map} ($r$= ', num2str(r) ,')'),'interpreter', 'latex', 'fontsize', 14)