% Created  by OctaveOliviers
%          on 2020-09-27 13:02:36
%
% Modified on 2020-09-30 10:58:11

% Experiment on random patterns, uniformly distributed

clear all
% clc
disp(" ")

% configure random number generator
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )
addpath( './data/' )



%%%%%%%%%%%%%%%%%%%%%%% set up experiment parameters %%%%%%%%%%%%%%%%%%%%%%

% patterns
num_pat = 100 ; % number of patterns to memorize
dim_pat = 5 ; % dimension of the patterns to memorize

% experiment parameters
num_test = 5 ;
step_siz = 10 ;
assert( mod(num_pat, step_siz)==0, "Make sure step_siz is a divisor of num_pat" )



%%%%%%%%%%%%% build models and store error as well as variance %%%%%%%%%%%%

% Hopfield trained with Hebbian rule
% hebb_err = NaN( num_pat/step_siz, 1) ;
% hebb_std = NaN( num_pat/step_siz, 1) ;

% Hopfield trained with Pseudo Inverse rule
% pinv = Hopfield_Network( 'pi' ) ;

% Modern Hopfield trained with Dense Associative Memory
% dam = Dense_Associative_Memory( img, 'repol', 4 ) ;

% Modern Hopfield trained with Transformer Memory
tm_perc         = zeros( num_pat/step_siz, 1) ;
tm_perc_var     = zeros( num_pat/step_siz, 1) ;

% C-LS-SVM model polynomial kernel degree 3
cls3_perc        = zeros( num_pat/step_siz, 1) ;
cls3_perc_var    = zeros( num_pat/step_siz, 1) ;

% C-LS-SVM model polynomial kernel degree 5
cls5_perc        = zeros( num_pat/step_siz, 1) ;
cls5_perc_var    = zeros( num_pat/step_siz, 1) ;

% deep C-LS-SVM model
dcls_perc        = zeros( num_pat/step_siz, 1) ;
dcls_perc_var    = zeros( num_pat/step_siz, 1) ;


%%%%%%%%%%%%% perform tests for increasing number of patterns %%%%%%%%%%%%%

% show progress bar
p = ProgressBar(num_pat/step_siz) ; 
tic
% ? add number of workers ?
parfor i = 1:num_pat/step_siz

    % Create models
    % Modern Hopfield trained with Transformer Memory
    tm = transformer_memory() ;
    % C-LS-SVM with polynomial kernel degree 3
    cls3 = clssvm_poly3(dim_pat) ;
    % C-LS-SVM with polynomial kernel degree 5
    cls5 = clssvm_poly5(dim_pat) ;
    % deep C-LS-SVM model
    dcls = deep_clssvm_3_poly3(dim_pat) ;
    
    for j = 1:num_test

        X = rand(dim_pat, i*step_siz) ;
        
%         % Hopfield trained with Hebbian rule
%         hebb = Hopfield_Network( X, 'h' ) ;
%         Y = hebb.simulate_one_step( X ) ;
%         [hebb_err(i), hebb_std(i)] = error_std( X, Y ) ;


        % Modern Hopfield trained with Transformer Memory
        % train model
        tm = tm.train( X ) ;
        % compute % of memories that are locally stable equilibria        
        [tm_perc(i), tm_perc_var(i)] = update_mean_var(tm_perc(i), tm_perc_var(i), perc_of_memories(tm), j) ;
        
        
        % C-LS-SVM with polynomial kernel degree 3
        % train model
        cls3 = cls3.train( X ) ;
        % compute % of memories that are locally stable equilibria
        [cls3_perc(i), cls3_perc_var(i)] = update_mean_var(cls3_perc(i), cls3_perc_var(i), perc_of_memories(cls3), j) ;
        
        
        % C-LS-SVM with polynomial kernel degree 3
        % train model
        cls5 = cls5.train( X ) ;
        % compute % of memories that are locally stable equilibria
        [cls5_perc(i), cls5_perc_var(i)] = update_mean_var(cls5_perc(i), cls5_perc_var(i), perc_of_memories(cls5), j) ;
            
        
%         % deep C-LS-SVM
%         % train model
%         dcls = dcls.train( X ) ;
%         % compute % of memories that are locally stable equilibria
%         [dcls_perc(i), dcls_perc_var(i)] = update_mean_var(dcls_perc(i), dcls_perc_var(i), perc_of_memories(dcls), j) ;

    end
    
    % update progress bar
    p.progress;
end
% finish progress bar
p.stop;
toc

% print out integral under curve
disp("Integral of average")
% Modern Hopfield trained with Transformer Memory
disp("      Transformer Hopfield : " + sum(tm_perc) )
% C-LS-SVM with polynomial kernel of degree 3
disp("      C-LS-SVM poly 3      : " + sum(cls3_perc) )
% C-LS-SVM with polynomial kernel of degree 5
disp("      C-LS-SVM poly 5      : " + sum(cls5_perc) )
% % deep C-LS-SVM
disp("      deep C-LS-SVM        : " + sum(dcls_perc) )
disp(" ")


%%
%%%%%%%%%% visualize % of patterns that are locally stable equi. %%%%%%%%%%

% visualize results
x = [ step_siz:step_siz:num_pat, fliplr(step_siz:step_siz:num_pat) ] ;
% hebb_inBetween = [ hebb_err' - hebb_std', fliplr( hebb_err' + hebb_std' ) ] ;
tm_inBetween = [ tm_perc' - sqrt(tm_perc_var/num_test)', fliplr( tm_perc' + sqrt(tm_perc_var/num_test)' ) ] ;
cls3_inBetween = [ cls3_perc' - sqrt(cls3_perc_var/num_test)', fliplr( cls3_perc' + sqrt(cls3_perc_var/num_test)' ) ] ;
cls5_inBetween = [ cls5_perc' - sqrt(cls5_perc_var/num_test)', fliplr( cls5_perc' + sqrt(cls5_perc_var/num_test)' ) ] ;
dcls_inBetween = [ dcls_perc' - sqrt(dcls_perc_var/num_test)', fliplr( dcls_perc' + sqrt(dcls_perc_var/num_test)' ) ] ;

figure('position', [100 100 600 300])
box on
hold on
%
% fill(x, 100*hebb_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_pat, 100*hebb_err, 'linewidth', 1, 'color', [0 0 0])
%
tm_ib = fill(x, tm_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_pat, tm_perc, 'linewidth', 1, 'color', [0 0 0])
%
cls3_ib = fill(x, cls3_inBetween, [210 255 210]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_pat, cls3_perc, 'linewidth', 1, 'color', [0 0 0])
%
cls5_ib = fill(x, cls5_inBetween, [255 210 210]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_pat, cls5_perc, 'linewidth', 1, 'color', [0 0 0])
%
% dcls_ib = fill(x, dcls_inBetween, [255 210 255]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_pat, dcls_perc, 'linewidth', 1, 'color', [0 0 0])
%
line([0 num_pat], [0, 0], 'color', [.1,.1,.1], 'linestyle', ':')
line([0 num_pat], [100, 100], 'color', [.1,.1,.1], 'linestyle', ':')
%
hold off
xlabel("Number of memories") ;
ylabel(["% of memories that are", "locally stable equilibria"]) ;
xlim([0, num_pat])
ylim([-10, 110])
yticks([0, 20, 40, 60, 80, 100])
legend( [tm_ib, cls3_ib, cls5_ib], {'Modern Hopfield Network', 'C-LS-SVM (poly 3)', 'C-LS-SVM (poly 5)'}, 'location', 'northeast')
title(["The capacity of the model grows", "with the dimension of the feature space"])

%% have models memorize additional pattern and check reconstruction error

clear all
% clc
disp(" ")
% configure random number generator
rng(10) ;

% patterns
num_pat = 200 ; % number of patterns to memorize
dim_pat = 10 ; % dimension of the patterns to memorize

% Create models
% Modern Hopfield trained with Transformer Memory
%    model = transformer_memory() ;
% C-LS-SVM with polynomial kernel degree 3
%    model = clssvm_poly3(dim_pat) ;
% C-LS-SVM with polynomial kernel degree 5
%    model = clssvm_poly5(dim_pat) ;
% C-LS-SVM with polynomial kernel degree 10
%    model = clssvm_poly10(dim_pat) ;
% C-LS-SVM with rbf kernel
    model = clssvm_rbf(dim_pat) ;
    

% std of gaussian noise
noise_std = 0.1 ;
max_update_steps = 100 ;

num_test = 10 ;
err_mean = 0 ;
err_var = 0 ;

% show progress bar
p = ProgressBar(num_test) ; 
tic

X = rand(dim_pat, num_pat) ;

for n = 1:num_test
    
    x = X(:, randin) 
    
    % additional pattern
    x = rand(dim_pat,1) ;

    % train model with additional memory
    model = model.train( X ) ;
    
    % initialize system
    x_kmo = x + noise_std*randn(size(x)) ;
    x_k = model.simulate_one_step(x_kmo) ;
    % update state until convergence
    for i = 1:max_update_steps
        % update state
        x_kpo = model.simulate_one_step(x_k) ;
        if norm(x_k - x_kpo) >= 0.95*norm(x_kmo - x_k)
            x_k = x_kpo ;
            % converged
            break
        end
        x_kmo = x_k ;
        x_k = x_kpo ;
    end
    
    [err_mean, err_var] = update_mean_var(err_mean, err_var, norm(x_k-x), n) ;
        
    % update progress bar
    p.progress;
end
% finish progress bar
p.stop;
toc

% disp("Stats for model " + model)
disp("     mean error = " + err_mean )
disp("     std error  = " + sqrt(err_var/num_test) )


%% error_std: compute % of error and its standard deviation
function [e, s] = error_std(X, Y)
    
    % assert( prod( size(X)==size(Y) ), "Size of input and output don't match")

    % percentage error on each individual pattern
    err_ind = sum( abs( X-Y )/2, 1 ) / size(X, 1) ;

    s = std(err_ind) ;
    e = mean(err_ind) ;
end


%% incrementally update mean and variance
% http://datagenetics.com/blog/november22017/index.html
function [new_m, new_var] = update_mean_var(old_m, old_var, new_data, test_num)
    
    % update mean
    new_m = old_m + (new_data-old_m) / test_num ;
    % update variance
    new_var = old_var + (new_data-old_m) * (new_data-new_m) ;

end


%% percentage_of_memories: compute % of memories that are locally stable equilibria
function perc = perc_of_memories(model)
    
    tol = 1e-1 ;

    % all memories whose error is smaller than the tolerance
    idx_err = vecnorm(model.model_error()) <= tol ;

    % memories in idx_err whose jacobian is stable
    idx_err_jac = NaN(sum(idx_err), 1) ;
    J = model.model_jacobian_3D(idx_err) ;
    for i = 1:sum(idx_err)
        idx_err_jac(i) = abs(eigs(J(:, :, i), 1, 'largestabs')) < 1 ;
    end

    perc = 100 * sum(idx_err_jac) / model.num_memories() ;

end