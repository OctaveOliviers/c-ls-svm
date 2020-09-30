% Created  by OctaveOliviers
%          on 2020-09-27 13:02:36
%
% Modified on 2020-09-30 10:58:11

% Experiment on MNIST data set

clear all
clc

% configure random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% load data
small = true ; % load small dataset
num_img = 200 ; % number of images to load
dim_img = 20 ; % dimension of the memories
pool_size = 5 ; % size of pooling (should be divider of 20)
% img = read_mnist(pool_size, num_img) ;
% img = rand(dim_img, num_img) ;
% siz_img = size(img, 1) ;


% experiment parameters
num_test = 20 ;
step_siz = 5 ;
assert( mod(num_img, step_siz)==0, "Make sure step_siz is a divisor of num_img")

% build models and store error as well as variance
% Hopfield trained with Hebbian rule
hebb_err = NaN( num_img/step_siz, 1) ;
hebb_std = NaN( num_img/step_siz, 1) ;

% Hopfield trained with Pseudo Inverse rule
% pinv = Hopfield_Network( 'pi' ) ;

% Modern Hopfield trained with Dense Associative Memory
% dam = Dense_Associative_Memory( img, 'repol', 4 ) ;

% Modern Hopfield trained with Transformer Memory
tm_err = NaN( num_img/step_siz, 1) ;
tm_std = NaN( num_img/step_siz, 1) ;
tm_perc = zeros( num_img/step_siz, 1) ;
tm_perc_var = zeros( num_img/step_siz, 1) ;

% C-LS-SVM model
cls_err = NaN( num_img/step_siz, 1) ;
cls_std = NaN( num_img/step_siz, 1) ;
cls_perc = NaN( num_img/step_siz, 1) ;
cls_perc_var = NaN( num_img/step_siz, 1) ;


for i = 1:num_img/step_siz

    for j = 1:num_test

%         X = img(:, 1:(i*step_siz)) ;
        X = rand(dim_img, i*step_siz) ;
        
%         % Hopfield trained with Hebbian rule
%         hebb = Hopfield_Network( X, 'h' ) ;
%         Y = hebb.simulate_one_step( X ) ;
%         [hebb_err(i), hebb_std(i)] = error_std( X, Y ) ;

        % Modern Hopfield trained with Transformer Memory
        tm = Transformer_Memory( X, 100 ) ;
        % Y = sign(tm.simulate_one_step( X )) ;
        % [tm_err(i), tm_std(i)] = error_std( X, Y ) ;
        perc_j = perc_of_memories(tm) ;

        % incremental mean 
        % http://datagenetics.com/blog/november22017/index.html
        mean_jm1 = tm_perc(i) ;
        tm_perc(i) = mean_jm1 + (perc_j - mean_jm1) / j ;

        % incremental variance
        tm_perc_var(i) = tm_perc_var(i) + (perc_j-mean_jm1) * (perc_j-tm_perc(i)) ;
    end
end


% visualize results
x = [ step_siz:step_siz:num_img, fliplr(step_siz:step_siz:num_img) ] ;
% hebb_inBetween = [ hebb_err' - hebb_std', fliplr( hebb_err' + hebb_std' ) ] ;
tm_inBetween = [ tm_perc' - sqrt(tm_perc_var/num_test)', fliplr( tm_perc' + sqrt(tm_perc_var/num_test)' ) ] ;

figure('position', [100 100 600 300])
box on
hold on
%
% fill(x, 100*hebb_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_img, 100*hebb_err, 'linewidth', 1, 'color', [0 0 0])
%
fill(x, tm_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_img, tm_perc, 'linewidth', 1, 'color', [0 0 0])
%
hold off
xlabel("Number of memories") ;
ylabel(["% of memories that are", "locally stable equilibria"]) ;
xlim([0, num_img])
ylim([1, 100])

%% error_std: compute % of error and its tandard deviation
function [e, s] = error_std(X, Y)
    
    % assert( prod( size(X)==size(Y) ), "Size of input and output don't match")

    % percentage error on each individual pattern
    err_ind = sum( abs( X-Y )/2, 1 ) / size(X, 1) ;

    s = std(err_ind) ;
    e = mean(err_ind) ;
end


%% percentage_of_memories: compute % of memories that are locally stable equilibria
function perc = perc_of_memories(model)
    
    tol = 1e-1 ;

    % all memories whose error is smaller than the tolerance
    idx_err = vecnorm(model.model_error()) <= tol ;

    % memories in idx_err whose jacobian is stable
    idx_err_jac = NaN(sum(idx_err), 1) ;
    J = model.model_jacobian(idx_err) ;
    for i = 1:sum(idx_err)
        idx_err_jac(i) = abs(eigs(J(:, :, i), 1, 'largestabs')) < 1 ;
    end

    perc = 100 * sum(idx_err_jac) / model.num_memories() ;

end