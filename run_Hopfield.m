% Created  by OctaveOliviers
%          on 2020-03-29 17:04:27
%
% Modified on 2020-05-19 15:55:19

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

% warning( 'Results are not correct.' )

% dimension of memories
N = 100 ;

% parameters of test
max_alpha = 20 ; % maximal ration of P/N times 100
num_test  = 50 ;
res = zeros(1, max_alpha) ;

% initialize random number generator
rng(10) ;


for i = 1:max_alpha
    i

    % number of memories
    P = ceil(i/100 * N) ;
    
    for t = 1:num_test
    
        % generate bipolar memories
        X = 2*randi( [0, 1] , N, P ) - 1 ;

        W = X * X' / N ;

        % one step Hopfield equation
        X_kp1 = sign( W * X ) ;

        % compute percentage of errors on each memory
        perc_err = sum( abs(X - X_kp1)/2 , 'all' ) /N /P ;

        % running average 
        res(1, i) = res(1, i) + ( perc_err - res(1, i) )/t ;
    end
end


figure( 'position', [100, 100, 300, 285] )
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
% colors of plot
orange   = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green    = [58, 148, 22]/255 ;
red      = [194, 52, 52]/255 ;
yellow   = [241, 194, 50]/255 ;
box on
grid on
hold on
plot(0.01:0.01:max_alpha*0.01, 100*res, 'color', KUL_blue, 'linewidth', 1) ;
line( [0.14, 0.14], [0, 1] , 'color', orange, 'linewidth' ,1)
hold off
set(gca, 'FontSize',12)
% xlim([1, ceil(max_dim/10)*10 ])
% ylim([ 1 10*max(res(1, :, :), [], 'all') ])
% xlim([1, 26])
% ylim([1, 1e5])
% xticks([1, 5, 10, 15, 20, 25])
% yticks([1, 1e1, 1e2, 1e3, 1e4 ])
% % h =legend( le , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12 )
% h =legend( {'$\eta / \lambda = 10^{-4}$', '$\eta / \lambda = 10^{-3}$', '$\eta / \lambda = 10^{-2}$'} , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12 )
% set(h,'color','none') %transparent
% ylim([0, 0.5])
xlabel( 'Parameter $\alpha = P/N$', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
ylabel( 'Percentage of errors', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
% breakyaxis( [0.05, 0.4] )

% title( {'The capacity of a network', 'strongly depends on its feature map'}, 'interpreter', 'latex', 'fontsize', 14 )



% % build model
% model = Hopfield_Network( 'sign' ) ;
% % train model
% model = model.train( patterns ) ;
% % visualize model
% model.visualize() ;