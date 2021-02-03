% Created  by OctaveOliviers
%          on 2020-05-12 17:07:41
%
% Modified on 2020-06-04 12:52:02

% Experiment to visualize the maximal number of equilibria in a C-LS-SVM

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of the tests
max_dim   = 1 ;
steps     = 5 ;
tol       = 1e-4 ;
num_tests = 5 ;

% ratios eta/lambda
eta_p_lam = [ 1e-5, 1e-4 , 1e-3, 1e-2 ] ; 

% hold mean and variance for each ratio
res = zeros( 2, 1 + floor(max_dim/steps), length(eta_p_lam) ) ;

% feature map or kernel function
fun = 'poly' ;
% parameter of feature map or kernel function
param  = [3, 1] ;

% maximum capacity
max_cap = zeros(1,  1 + floor(max_dim/steps) ) ;


for r = 1:length(eta_p_lam)

    for d = 0:floor(max_dim/steps)
        d

        max_num = dimension_feature_space( 1+d*steps, fun, param ) ;
        max_cap(d+1) = max_num ;

        for t = 1:num_tests

            for n = max_num:-randi([5, 10]):1

                % generate uniform dataset between -1 and 1
                X = 2*rand(1+d*steps, n) - 1 ;

                PTP = phiTphi( X, X, fun, param ) ;

                err = max( vecnorm( X * inv( PTP + eta_p_lam(r)*eye(n) ) * PTP - X ) ) ;

                if err <= tol
                    break
                end
            end

            mean_prev = res(1, d+1, r) ;
            % running average
            res(1, d+1, r) = mean_prev + ( n - mean_prev )/t ;
            % running variance
            res(2, d+1, r) = res(2, d+1, r) + ( n - mean_prev )*( n - res(1, d+1, r) ) ;
        end
    end
end

% % generate bipolar data set recursively
% data = [ 1 -1 ] ;
% for d = 2:dim
%     data = [ data , data ; ones(1, 2^(d-1)), -ones(1, 2^(d-1)) ] ;
% end

le = create_legend( eta_p_lam ) ;

figure( 'position', [100, 100, 300, 285] )
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
% colors of plot
orange   = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green    = [58, 148, 22]/255 ;
red      = [194, 52, 52]/255 ;
yellow   = [241, 194, 50]/255 ;
colors   = [  KUL_blue ; yellow  ; red ] ;
style    = { '-o', '-+', '-d', '-x', '-s'} ;
box on
grid on
hold on

for r = 2:length(eta_p_lam) %1:length(eta_p_lam)
    errorbar( 1:steps:(max_dim+1), res(1, :, r), sqrt(res(2, :, r)/num_tests), style{r}, 'linewidth', 1, 'color', colors(r-1, :) ) ;
end

% maximum capacity
plot( 1:steps:(max_dim+1), max_cap, ':', 'color', KUL_blue )

hold off
set(gca, 'YScale', 'log', 'FontSize',12)
% xlim([1, ceil(max_dim/10)*10 ])
% ylim([ 1 10*max(res(1, :, :), [], 'all') ])
xlim([1, 26])
ylim([1, 1e5])
xticks([1, 5, 10, 15, 20, 25])
yticks([1, 1e1, 1e2, 1e3, 1e4 ])
ax = gca;
ax.YGrid = 'on';
ax.YMinorGrid = 'off';
h =legend( {'$\eta / \lambda = 10^{-4}$', '$\eta / \lambda = 10^{-3}$', '$\eta / \lambda = 10^{-2}$'} , 'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12 )
% set(h,'color','none') %transparent
xlabel( 'Number of neurons $N$', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
ylabel( 'Number of equilibria', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
% title( {'The capacity of a network', 'strongly depends on its feature map'}, 'interpreter', 'latex', 'fontsize', 14 )



%% Compute maximal capacity of different feature maps

max_dim = 100 ;
steps   = 10 ; 

assert( rem(max_dim, steps) == 0 , 'Maxdim should be a multiple of steps' )

phis    = { 'tanh', 'sign', 'poly', 'poly', 'poly' } ;
thetas  = { 0, 0, [3, 1], [5, 1], [7, 1] } ;

max_cap_feat = zeros( 1 + floor(max_dim/steps), length(phis) ) ;

for p = 1:length(phis)
    for d = 0:floor(max_dim/steps)
        max_cap_feat( d+1 , p ) = dimension_feature_space( 1+d*steps, phis{p}, thetas{p} ) ;
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
colors   = [ red ; orange; yellow ; green ; KUL_blue ] ;
style    = { '-o', '-+', '-d', '-x', '-s'} ;

box on
grid on
hold on
for p = length(phis):-1:1
    plot( 1:steps:(max_dim+1), max_cap_feat(:, p), style{p}, 'linewidth', 1, 'color', colors(p, :) ) ;
end
hold off
xticks([1, 25, 50, 75, 100])
ylim( [ 1 100*max(max_cap_feat, [], 'all') ] )
set(gca, 'YScale', 'log', 'FontSize',12)
legend( {'poly-7', 'poly-5', 'poly-3', 'sign', 'tanh'}, 'location', 'best', 'interpreter', 'latex', 'fontsize', 12 )
xlabel( 'Number of neurons $N$', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
ylabel( 'Number of equilibria', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
% title( {'The capacity of a network', 'strongly depends on its feature map'}, 'interpreter', 'latex', 'fontsize', 14 )


function d = dimension_feature_space( input_dim, fun, param )
    % input_dim     dimension of input space
    % fun           feature map or kernel function
    % param         parameter of feature map or kernel function

    switch lower(fun)
        case 'tanh'
            d = input_dim ;

        case 'sign'
            d = input_dim ;

        case {'poly', 'polynomial', 'p'}
            if length(param) == 1
                deg = param ;
            elseif length(param) == 2
                deg = param(1) ;
            else
                error( "Did not understand parameters of polynomial kernel." )
            end

            d = nchoosek( input_dim + deg, deg ) ; % / (input_dim+1) ;

        otherwise
            error( "Did not understand which feature map or kernel function." )
    end

end


function l = create_legend( ratios )

    % l = zeros(1, length(ratios)) ;

    for r = 1:length(ratios)

        l(r) = strcat( "$\eta / \lambda = 10^{", num2str(log10(ratios(r))), "}$") ;

    end

end