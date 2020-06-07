% Created  by OctaveOliviers
%          on 2020-05-03 11:21:50
%
% Modified on 2020-05-03 16:10:07

clear all
clc
rng(10)

% generate data set
prec   = 1 ;
scale  = 5 ;
x      = -scale:prec:scale ;
y_true = sin(x) ;
y_corr = y_true + 0.5*randn(size(y_true)) ;


% overfitting model
p_over = polyfit( x, y_corr, 10 ) ;

% overfitting model
p_under = polyfit( x, y_corr, 3 ) ;

orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;

% show overfitting
figure('position', [100, 100, 300, 270])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
% grid on ;
box on ;
hold on ;
true_fun = plot(-scale:0.1:scale, sin(-scale:0.1:scale), '--', 'color', [0.5, 0.5,0.5]) ;
eval_fun = plot( -scale:0.1:scale, polyval(p_over, -scale:0.1:scale) , 'color', KUL_blue, 'linewidth', 1 ) ;
data_cor = plot(x, y_corr, '.', 'color', orange, 'markersize', 12) ;
hold off ;
xlim([-scale scale])
ylim([-2 5])
set(gca,'FontSize',12)
xlabel('Input $x$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('Output $y$', 'interpreter', 'latex', 'fontsize', 14)
legend( [true_fun, data_cor, eval_fun], {"True function", "Noisy data", "Estimated function"} , 'location', 'north', 'interpreter', 'latex', 'fontsize', 12)
title( strcat('\textbf{Overfitting} (', num2str(length(p_over)) ,' parameters)'),'interpreter', 'latex', 'fontsize', 14)


% show underfitting
figure('position', [100, 100, 300, 270])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
% grid on ;
box on ;
hold on ;
true_fun = plot(-scale:0.1:scale, sin(-scale:0.1:scale), '--', 'color', [0.5, 0.5,0.5]) ;
eval_fun = plot( -scale:0.1:scale, polyval(p_under, -scale:0.1:scale) , 'color', KUL_blue, 'linewidth', 1 ) ;
data_cor = plot(x, y_corr, '.', 'color', orange, 'markersize', 12) ;
hold off ;
xlim([-scale scale])
ylim([-2 5])
set(gca,'FontSize',12)
xlabel('Input $x$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('Output $y$', 'interpreter', 'latex', 'fontsize', 14)
legend( [true_fun, data_cor, eval_fun], {"True function", "Noisy data", "Estimated function"} , 'location', 'north', 'interpreter', 'latex', 'fontsize', 12)
title( strcat('\textbf{Underfitting} (', num2str(length(p_under)) ,' parameters)'),'interpreter', 'latex', 'fontsize', 14)