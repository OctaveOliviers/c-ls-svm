% Created  by OctaveOliviers
%          on 2020-05-04 21:41:11
%
% Modified on 2020-05-04 22:50:55

clear all
clc

num = 5 ;
typ = 'med' ;

% create data set
switch typ
    case 'pos'
        X = 1:num ;
    case 'neg'
        X = -num:-1 ;
    otherwise
        X = -floor(num/2):floor(num/2);
end

phi   = @(x) [ x ; x.^2 ] ;
d_phi = @(x) [ 1 ; x ] ;

S = zeros(2, 2) ;
for x = X
    S = S + d_phi(x)*d_phi(x)' ;
end


orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;

% visualize points in feature space
figure('position', [100, 100, 170, 160])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
grid on ;
box on ;
hold on ;
% plot feature map
x = (min(X)-1):0.1:(max(X)+1) ;
phi_x = phi(x) ;
line_phi = plot( phi_x(1, :), phi_x(2, :), 'color', KUL_blue, 'linewidth', 1.5 ) ;
% plot points in feature space
phi_X = phi(X) ;
pnts = plot( phi_X(1, :), phi_X(2, :), '.', 'color', orange, 'MarkerSize', 20 ) ;
hold off ;
ylim([ min(phi_X(2, :))-2 max(phi_X(2, :))+2])
% axis equal
xticks([ sort( unique(phi_X(1, :))) ])
yticks([ sort( unique(phi_X(2, :))) ])
set(gca,'FontSize',12)
xlabel('$\varphi_1(x)$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$\varphi_2(x)$', 'interpreter', 'latex', 'fontsize', 14)
% legend( [line_log, point_start, line_path], {"$r \; x^{(k)} \; (1-x^{(k)})$", "Starting point", "Movement"} , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12)
% title( strcat('\textbf{Logistic map} ($r$= ', num2str(r) ,')'),'interpreter', 'latex', 'fontsize', 14)


% visualize ellipse
lim = 10 ;
figure('position', [100, 100, 170, 160])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
grid on ;
box on ;
hold on ;
% plot covariance matrix
h = error_ellipse( S )
set( h, 'LineWidth', 2, 'Color', orange )
hold off ;
xlim([ -lim lim ])
ylim([ -lim lim ])
axis equal
xticks([ -lim 0 lim ])
yticks([ -lim 0 lim ])
set(gca,'FontSize',12)
xlabel('$w_1$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$w_2$', 'interpreter', 'latex', 'fontsize', 14)
% legend( [line_log, point_start, line_path], {"$r \; x^{(k)} \; (1-x^{(k)})$", "Starting point", "Movement"} , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12)
% title( strcat('\textbf{Logistic map} ($r$= ', num2str(r) ,')'),'interpreter', 'latex', 'fontsize', 14)