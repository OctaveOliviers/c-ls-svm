% Created  by OctaveOliviers
%          on 2020-04-26 18:36:03
%
% Modified on 2020-04-27 14:47:56

% Simulate the logistic map
% http://physics.ucsc.edu/~peter/242/logistic.pdf

clear all
clc

% number of points to simulate
num_sim = 4 ;
% number of steps to simulate over
len_sim = 200 ;
% store path in the state space
path = zeros(2, len_sim+1, num_sim) ;

% parameter
b  = 0.6 ;
dt = 1e-1 ;
% Pendulum discretization
map = @(x) [  x(1, :)         + dt*x(2, :) ; ...
             -dt*sin(x(1, :)) + (1-b*dt)*x(2, :) ] ;

% points to start simulation
prec  = 0.8 / (num_sim-1) ;
% path(:, 1) = 0.1:prec:0.9 ;
path(:, 1, :) = [ -4 0 0 4 ; 2 3 -3 -2] ;

% simulate
for i = 1:len_sim
    path(:, i+1, :) = map( path(:, i, :) ) ;
end

orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;

% visualize path
figure('position', [100, 100, 300, 285])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
box on ;
grid on ;
hold on ;
% vector field
wdw = 4 ; % window
prec = wdw/5 ;
x = -wdw:prec:wdw ;
y = -wdw:prec:wdw ;
[X, Y] = meshgrid(x, y) ;           
%
F = map( [ X(:)' ; Y(:)' ] ) ;
f1 = reshape( F(1, :), [length(x), length(y)] ) ;
f2 = reshape( F(2, :), [length(x), length(y)] ) ;
scale = 1 ;
vector = quiver( X, Y, (f1-X), (f2-Y), scale, 'color', KUL_blue ) ;
%
for n = 1:num_sim
    line_path   = plot( squeeze( path(1, :, n) ), squeeze( path(2, :, n) ), 'color', orange, 'linewidth', 1 ) ;
    point_start = plot( path(1, 1, n), path(2, 1, n), 'o', 'color', orange, 'linewidth', 1.5 ) ;
end
% equilibria
plot(0,0,'.', 'color',green, 'markersize', 25)
plot(-pi,0,'.', 'color',red, 'markersize', 25)
plot(pi,0,'.', 'color',red, 'markersize', 25)
hold off ;
% ax.XAxisLocation = 'origin';
axis equal
xlim([-4, 4]) ;
xticks([-pi 0 pi])
xticklabels({'-$\pi$', '0', '$\pi$'})
yticks([-pi 0 pi])
yticklabels({'-$\pi$', '0', '$\pi$'})
set(gca,'FontSize',12)
xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)
legend( [vector, point_start, line_path], {"Vector field", "Starting point", "Movement"} , 'location', 'southwest', 'interpreter', 'latex', 'fontsize', 12)
title( strcat('\textbf{Pendulum} ($b$=',num2str(b) ,', $\Delta t$=',num2str(dt) ,')'),'interpreter', 'latex', 'fontsize', 14)