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
num_img = 200 ; % number of images to load
dim_img = 20 ; % dimension of the memories

% experiment parameters
num_test = 5 ;
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
% parameters
beta = 20 ;
% keep track of % of memories that are locally stable equilibria
tm_perc         = zeros( num_img/step_siz, 1) ;
tm_perc_var     = zeros( num_img/step_siz, 1) ;

% C-LS-SVM model
% parameters
space           = 'dual' ;          % space to train layer
dim_input       = dim_img ;         % dimension of the input space
hp_equi         = 1e2 ;             % importance of equilibrium objective
hp_stab         = 1e0 ;             % importance of local stability objective
hp_reg          = 1e-2 ;            % importance of regularization
feat_map        = 'poly' ;           % chosen feature map or kernel function
feat_map_param  = [3, 1] ;             % parameter of feature map or kernel function
% keep track of % of memories that are locally stable equilibria
cls_perc        = zeros( num_img/step_siz, 1) ;
cls_perc_var    = zeros( num_img/step_siz, 1) ;


% deep C-LS-SVM model
% parameters
spaces          = {'primal', 'primal', 'primal' } ;
dim_inputs       = {dim_img, dim_img, dim_img} ;
hp_equis         = {1e2, 1e2, 1e2} ;
hp_stabs         = {1e0, 1e0, 1e0} ;
hp_regs          = {1e-2, 1e-2, 1e-2} ;
feat_maps        = {'poly', 'poly', 'poly'} ;
feat_maps_param  = {[3, 1], [3, 1], [3, 1]} ;
% keep track of % of memories that are locally stable equilibria
dcls_perc        = zeros( num_img/step_siz, 1) ;
dcls_perc_var    = zeros( num_img/step_siz, 1) ;


for i = 1:num_img/step_siz

    for j = 1:num_test

        X = rand(dim_img, i*step_siz) ;
        
%         % Hopfield trained with Hebbian rule
%         hebb = Hopfield_Network( X, 'h' ) ;
%         Y = hebb.simulate_one_step( X ) ;
%         [hebb_err(i), hebb_std(i)] = error_std( X, Y ) ;


        % Modern Hopfield trained with Transformer Memory
        % create model
        tm = Transformer_Memory( X, beta ) ;
        % compute % of memories that are locally stable equilibria
        perc_j = perc_of_memories(tm) ;
        % incremental mean
        mean_jm1 = tm_perc(i) ;
        tm_perc(i) = mean_jm1 + (perc_j - mean_jm1) / j ;
        % incremental variance
        tm_perc_var(i) = tm_perc_var(i) + (perc_j-mean_jm1) * (perc_j-tm_perc(i)) ;
        
        
        % C-LS-SVM
        % create model
        cls = CLSSVM( ) ;
        cls = cls.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;
        cls = cls.train( X ) ;
        % compute % of memories that are locally stable equilibria
        perc_j = perc_of_memories(cls) ;
        % incremental mean
        mean_jm1 = cls_perc(i) ;
        cls_perc(i) = mean_jm1 + (perc_j - mean_jm1) / j ;
        % incremental variance
        cls_perc_var(i) = cls_perc_var(i) + (perc_j-mean_jm1) * (perc_j-cls_perc(i)) ;
        
        
%         % deep C-LS-SVM
%         % create model
%         dcls = CLSSVM( ) ;
%         for l = 1:length(feat_maps)
%             dcls = dcls.add_layer( spaces{l}, dim_inputs{l}, hp_equis{l}, hp_stabs{l}, hp_regs{l}, feat_maps{l}, feat_maps_param{l} ) ;
%         end
%         dcls = dcls.train( X ) ;
%         % compute % of memories that are locally stable equilibria
%         perc_j = perc_of_memories(dcls) ;
%         % incremental mean
%         mean_jm1 = dcls_perc(i) ;
%         dcls_perc(i) = mean_jm1 + (perc_j - mean_jm1) / j ;
%         % incremental variance
%         dcls_perc_var(i) = dcls_perc_var(i) + (perc_j-mean_jm1) * (perc_j-dcls_perc(i)) ;
    end
end


% visualize results
x = [ step_siz:step_siz:num_img, fliplr(step_siz:step_siz:num_img) ] ;
% hebb_inBetween = [ hebb_err' - hebb_std', fliplr( hebb_err' + hebb_std' ) ] ;
tm_inBetween = [ tm_perc' - sqrt(tm_perc_var/num_test)', fliplr( tm_perc' + sqrt(tm_perc_var/num_test)' ) ] ;
cls_inBetween = [ cls_perc' - sqrt(cls_perc_var/num_test)', fliplr( cls_perc' + sqrt(cls_perc_var/num_test)' ) ] ;
% dcls_inBetween = [ dcls_perc' - sqrt(dcls_perc_var/num_test)', fliplr( dcls_perc' + sqrt(dcls_perc_var/num_test)' ) ] ;

figure('position', [100 100 600 300])
box on
hold on
%
% fill(x, 100*hebb_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_img, 100*hebb_err, 'linewidth', 1, 'color', [0 0 0])
%
tm_ib = fill(x, tm_inBetween, [210 210 255]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_img, tm_perc, 'linewidth', 1, 'color', [0 0 0])
%
cls_ib = fill(x, cls_inBetween, [210 255 210]/255 , 'EdgeColor', [1 1 1]);
plot(step_siz:step_siz:num_img, cls_perc, 'linewidth', 1, 'color', [0 0 0])
% %
% dcls_ib = fill(x, dcls_inBetween, [255 210 210]/255 , 'EdgeColor', [1 1 1]);
% plot(step_siz:step_siz:num_img, dcls_perc, 'linewidth', 1, 'color', [0 0 0])
%
hold off
xlabel("Number of memories") ;
ylabel(["% of memories that are", "locally stable equilibria"]) ;
xlim([0, num_img])
ylim([-10, 110])
yticks([0, 20, 40, 60, 80, 100])
% legend( [tm_ib, cls_ib, dcls_ib], {'Transformer memory', 'C-LS-SVM', 'Deep C-LS-SVM'}, 'location', 'northeast')



%% error_std: compute % of error and its standard deviation
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
    J = model.model_jacobian_3D(idx_err) ;
    for i = 1:sum(idx_err)
        idx_err_jac(i) = abs(eigs(J(:, :, i), 1, 'largestabs')) < 1 ;
    end

    perc = 100 * sum(idx_err_jac) / model.num_memories() ;

end