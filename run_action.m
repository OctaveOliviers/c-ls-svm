% Created  by OctaveOliviers
%          on 2020-03-29 17:04:35
%
% Modified on 2020-06-02 19:57:08

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

load('data/hello_written.mat') ; 
dim_movements = size(z, 1) ; 
num_movements = 1 ; 
len_movements = size(z, 2) ;

% create patterns to memorize

% initialize random number generator
rng(10)
% movements = 2*randn(dim_movements, num_movements, len_movements) ;
% store 4 movements
movements = zeros(dim_movements, num_movements, len_movements) ;
movements(:, 1, :) =  z + [1; -1.5] ;
% movements(:, 2, :) = -z + [-1; -1] ; 
% movements(:, 3, :) = -z + [6; 3.5] ; 
% movements(:, 4, :) =  z + [-6; 3] ;


% model architecture
formulation = 'dual' ;
feature_map = 'rbf' ;
parameter   = 2 ;
p_err = 1e2  ;  % importance of error
p_reg = 1e-2 ;  % importance of regularization
p_drv = 1e1  ;  % importance of minimizing derivative
% num_layers  = len_movements-1 ;



%% build model to memorize patterns

model = CLSSVM( ) ;
% add one layer for each step of the movement
for l = 1:len_movements-1
    model = model.add_layer( formulation, dim_movements, p_err, p_drv, p_reg, feature_map, parameter ) ;
end



%% train model

% train each layer
for l = 1:len_movements-1
    model.layers{l} = model.layers{l}.train( movements(:, :, l), movements(:, :, l+1) ) ;
end



%% simulate model

% number of points to simulate for each movement
num_sim = 10 ;
% starting point of each simulation
start_sim = repmat( movements( :, :, 1 ), 1, num_sim ) + 0.5 * randn( dim_movements, num_sim*num_movements ) ;
% store reconstructed movement
recon = zeros(dim_movements, num_sim*num_movements, len_movements) ;
recon(:, :, 1) = start_sim ;
% simulate
for l = 2:len_movements
    for p = 1:num_sim*num_movements
        % simulate step stored in layer l
        recon(:, p, l) = model.layers{l-1}.simulate_one_step( recon(:, p, l-1) ) ;
    end
end



%% visualize continuous movement
% colors
orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;

gran = 200 ;

t = linspace( 0, 1, len_movements ) ;
xt = linspace( 0, 1, gran ) ;
xx = spline( t, squeeze(movements(1, 1, :)), xt);

yt = linspace( 0, 1, gran ) ;
yy = spline( t, squeeze(movements(2, 1, :)), yt);

figure('position', [500, 700, 300, 240])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
box on
hold on
plot( xx, yy, 'color', KUL_blue, 'linewidth', 1, 'linestyle', '-' )
% plot( x, y, 'x', 'MarkerSize', 8, 'LineWidth', 2, 'color', KUL_blue )
hold off
xlim([0.5, 6.5])
% ylim([-3, 0.5])
ylim([-3.5, 1])
set(gca,'FontSize',12)
xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)



%% visualize discretized movement

figure('position', [500, 700, 300, 240])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
box on
hold on
plot( squeeze(movements(1, 1, :)), squeeze(movements(2, 1, :)), 'color', KUL_blue, 'linewidth', 1, 'linestyle', '-' )
hold off
xlim([0.5, 6.5])
% ylim([-3, 0.5])
ylim([-3.5, 1])
set(gca,'FontSize',12)
xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)



%% visualize simulations

% figure('position', [500, 700, 600, 240])
figure('position', [500, 700, 300, 240])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
box on
hold on

% % plot the original movements
% for p = 1:num_movements
%     l_original = plot( squeeze( movements(1, p, :) ), squeeze( movements(2, p, :) ), 'color', KUL_blue, 'linewidth', 1, 'linestyle', '-' ) ;
% end 

plot( xx, yy, 'color', KUL_blue, 'linewidth', 1, 'linestyle', '-' )

plot( -xx , -yy - 2.5, 'color', KUL_blue, 'linewidth', 1, 'linestyle', '-' )

% plot the reconstructed movements
for p = 1:num_sim*num_movements
    l_recon = plot( squeeze( recon(1, p, :) ), squeeze( recon(2, p, :) ), 'color', orange, 'linewidth', 1.5, 'linestyle', ':' ) ;
    p_start = plot( recon(1, p, 1), recon(2, p, 1), '.', 'MarkerSize', 20, 'color', orange ) ;
    p_end   = plot( recon(1, p, end), recon(2, p, end), 'x', 'MarkerSize', 8, 'LineWidth', 2, 'color', orange ) ;
end

hold off
xlim([0.5, 6.5])
ylim([-3.5, 1])
% xlim([-6.5, 6.5])
% ylim([-3.5, 1])
set(gca,'FontSize',12)
xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)
% legend( [l_original, l_recon, p_start, p_end], ...
%         {'Stored movement', 'Reconstructed movement', 'Start', 'End'}, ...
%         'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;


