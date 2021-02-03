% Created  by OctaveOliviers
%          on 2020-09-21 13:49:26
%
% Modified on 2020-09-27 12:28:52

% Experiment on Omniglot data set

clear all
clc

% configure random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% load data
small = true ;  % load small dataset
num_img = 500 ;  % number of images to load
pool_size = 5 ; % size of pooling (should be divider of 15)
% img = read_omniglot(small, num_img, pool_size) ;
img = 2*randi([0, 1], 10^2, num_img)-1 ;
siz_img = size(img, 1) ;


% experiment parameters
% num_test = 1 ;
step_siz = 5 ;
assert( mod(num_img, step_siz)==0, "Make sure step_siz is a divisor of num_img")

% build models and store error as well as variance
% Hopfield trained with Hebbian rule
hebb_err = NaN( num_img/step_siz, 1) ;
hebb_std = NaN( num_img/step_siz, 1) ;

% pinv = Hopfield_Network( 'pi' ) ;
% Modern Hopfield trained with Dense Associative Memory
% dam = Dense_Associative_Memory( img, 'repol', 4 ) ;
% Modern Hopfield trained with Transformer Memory
tm_err = NaN( num_img/step_siz, 1) ;
tm_std = NaN( num_img/step_siz, 1) ;


for i = 1:num_img/step_siz

    X = img(:, 1:(i*step_siz)) ;
    
    % Hopfield trained with Hebbian rule
    hebb = Hopfield_Network( X, 'h' ) ;
    Y = hebb.simulate_one_step( X ) ;
    [hebb_err(i), hebb_std(i)] = error_std( X, Y ) ;

%     % Modern Hopfield trained with Transformer Memory
%     tm = Transformer_Memory( X, 0.5 ) ;
%     Y = sign(tm.simulate_one_step( X )) ;
%     [tm_err(i), tm_std(i)] = error_std( X, Y ) ;

end

% visualize results
x = [ step_siz:step_siz:num_img, fliplr(step_siz:step_siz:num_img) ] ;
hebb_inBetween = [ hebb_err' - hebb_std', fliplr( hebb_err' + hebb_std' ) ] ;
% tm_inBetween = [ tm_err' - tm_std', fliplr( tm_err' + tm_std' ) ] ;

figure('position', [100 100 600 300])
box on
hold on
%
fill(x, 100*hebb_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_img, 100*hebb_err, 'linewidth', 1, 'color', [0 0 0])
%
% fill(x, 100*tm_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_img, 100*tm_err, 'linewidth', 1, 'color', [0 0 0])
%
hold off
xlabel("Number of memories") ;
ylabel("Percentage reconstruction error") ;
xlim([0, num_img])
ylim([1, 100])

%% error_std: compute % of error and its standard deviation
function [e, s] = error_std(X, Y)
    
    % assert( prod( size(X)==size(Y) ), "Size of input and output don't match")

    % percentage error on each individual pattern
    err_ind = sum( abs( X-Y )/2, 1 ) / size(X, 1) ;

    s = std(err_ind) ;
    e = mean(err_ind) ;
end