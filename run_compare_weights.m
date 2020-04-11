% Created  by OctaveOliviers
%          on 2020-03-29 17:04:32
%
% Modified on 2020-04-11 14:14:55

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

max_dim         = 10 ;
max_num         = 20 ;
num_tests       = 10 ;

% matrix that holds the difference between weight matrices
D = zeros(max_dim, max_num) ;

for n = 1:max_dim
    n

    P = find( (1:max_num) <= 2^n );

    model = Memory_Model_Shallow_Dual('poly', [1, 0], n^2, n/2, n/2) ;

    for p = P
        for i = 1:num_tests
            X = 2*randi([0, 1], [n, p]) - 1 ;

            % Hopfield's weights
            W_hopfield = X*X' ;

            % Model's weights
            model = model.train( X ) ;
            W_model = model.W ;

            % hold running average
            D(n, p) = D(n, p) + (norm( W_hopfield - W_model , 'fro') - D(n, p))/i ;
        end        
    end
end

figure( 'position', [100, 100, 500, 500],'DefaultAxesFontSize', 12 )
box on
hold on
contour(D')
plot(1:max_dim, 2.^[1:max_dim], 'linestyle', '--', 'linewidth', 1) ;
hold off
xlim( [1, max_dim] )
ylim( [1, max_num] )
% legend( [flip(names), 'Hopfield', ], 'location', 'best', 'FontSize', 12 )
xlabel( 'Network size N', 'FontSize', 12 )
ylabel( 'Number of Patterns', 'FontSize', 12 )
title( 'Linear scale', 'FontSize', 12 )