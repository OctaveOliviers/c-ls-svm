% Created  by OctaveOliviers
%          on 2020-03-29 17:04:32
%
% Modified on 2020-05-12 14:36:00

% Compare the capacity of several feature maps

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './util/' )

max_dim         = 5 ;
tol             = 0.05 ;
num_tests       = 1 ;

% feature maps to test
spaces          = { 'primal', 'primal', 'dual',         'dual' } ;
phis            = { 'tanh', 'sign', 'poly',         'poly' } ;
names           = { 'tanh', 'sign', 'poly (deg 3)', 'poly (deg 5)' } ;
thetas          = { 0,      0,      [3, 1],         [5, 1] } ;
num_layers      = { 1,      1,      1,              1 } ;

data            = cell( length(phis), 4 ) ;
[data{:, 1}]    = deal( spaces ) ;
[data{:, 2}]    = deal( phis ) ;
[data{:, 3}]    = deal( thetas ) ;
[data{:, 4:5}]  = deal( zeros(1, max_dim) ) ;

cap_hopfld      = 1 + 0.138*[1:max_dim] ;

for d = 1:max_dim
    d
    for p = 1:length(phis)
        for i = 1:num_tests
            % capacity until recall error is larger than tol
            cap             = run_one_test( num_layers{p}, spaces{p}, phis{p}, thetas{p}, d, tol );

            mean_prev       = data{ p, 4 }(d) ;
            % running average
            data{ p, 4 }(d) = mean_prev + ( cap - mean_prev )/i ;
            % running variance
            data{ p, 5 }(d) = data{ p, 5 }(d) + ( cap - mean_prev )*( cap - data{ p, 4 }(d) ) ;
        end
    end
end


figure( 'position', [100, 100, 600, 285] )
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
% subplot(1, 2, 1)
% box on
% hold on
% for p = length(phis):-1:1
%     errorbar(1:max_dim, data{p, 4}, sqrt(data{p, 5}/num_tests), 'linewidth', 1) ;
% end
% plot(1:max_dim, cap_hopfld, 'linestyle', '-', 'linewidth', 1) ;
% hold off
% legend( [flip(names), 'Hopfield', ], 'location', 'best', 'FontSize', 12 )
% xlabel( 'Network size N', 'FontSize', 12 )
% ylabel( 'Network capacity P_c', 'FontSize', 12 )
% title( 'Linear scale', 'FontSize', 12 )
%
% subplot(1, 2, 2)
box on
hold on
for p = length(phis):-1:1
    errorbar(1:max_dim, data{p, 4}, sqrt(data{p, 5}/num_tests), 'linewidth', 1) ;
end
plot(1:max_dim, cap_hopfld, 'linestyle', '-', 'linewidth', 1) ;
plot(1:max_dim, 2.^[1:max_dim], 'linestyle', ':', 'linewidth', 1) ;
hold off
set(gca, 'YScale', 'log', 'FontSize',12)
legend( [flip(names), 'Hopfield'], 'location', 'eastoutside', 'interpreter', 'latex', 'fontsize', 12 )
xlabel( 'Number of neurons $N$', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
ylabel( 'Network capacity $P_c$', 'FontSize', 14, 'interpreter', 'latex', 'fontsize', 14 )
title( {'The capacity of a network', 'strongly depends on its feature map'}, 'interpreter', 'latex', 'fontsize', 14 )
%
% suptitle( {'The capacity of a network strongly depends on its feature map'} )



function cap = run_one_test( num_layers, space, fun, param, dim, tol )
    % num_layers    number of layers in network to test
    % space         whether to train in primal or dual space
    % fun           feature map to use for test
    % param         parameter of feature map
    % dim           dimension of network to evaluate
    % tol           maximal error on data point

    % create a data set with half of max possible number memories
    data      = 2*randi( [0, 1], dim, 2^(dim-1) ) - 1 ;
    err_rate  = 0 ;

    % create model
    p_err     = 1e2 ; % importance of error
    p_reg     = 1e-2 ; % importance of regularization
    p_drv     = 1e2 ; % importance of minimizing derivative

    % create model
    model = Memory_Model( ) ;
    model = model.add_layer( space, dim, p_err, p_drv, p_reg, fun, param ) ;

    for d = 1:2^dim

        % append new data point if capacity is larger than half the max number of 
        if d == 2^(dim-1) + 1
            data = [ data , 2*randi( [0, 1], dim, 2^(dim-1) ) - 1 ] ;
        end

        % train model
        model = model.train( data(:, 1:d) ) ;

        % stop if error on one pattern is too large
        % err = model.E 
        % vecnorm(err)
        err = data(:, 1:d) - sign( model.simulate_one_step(data(:, 1:d)) ) ;
        if ( max( vecnorm(err), [], 'all' ) >= tol )
            break
        end

        % stop if one pattern is not (locally) stable
        % [~, S] = model.jacobian_singular_values( ) ;
        % if ( max( S, [], 'all' ) >= 1 )
        %     break
        % end

    end

    cap = d ; %size(data, 2)-1 ;
    
end