% Created  by OctaveOliviers
%          on 2020-03-29 17:06:12
%
% Modified on 2020-05-09 08:23:56

clear all
clc
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of data set
siz_train       = 50 ;
siz_load        = 10000 ;
siz_val         = 100 ;
siz_test        = 1000 ; 
num_protos      = floor(siz_train/10) ;       % number of prototypes to select for each label
% parameters of model
num_layers      = 1 ;
formulation     = { 'dual' } ; 
dim_layer       = { 400 } ;
feature_map     = { 'rbf' } ; 
parameter       = { 50 } ;
p_err           = { 1e3 } ;  % importance of error
p_reg           = { 1e-2 } ;  % importance of regularization200
p_drv           = { 1e1 } ;  % importance of minimizing derivative
% parameters of training
k_cross_val     = 10 ;
% parameters of testing
tolerance       = 1e-3 ;
num_to_avg_over = 1 ;


%% Load dataset
% each is a b&w image of 20 x 20 pixels
% train_img (i, j, k) : i <= 20,    j <= 20, k <= 60000
% test_img  (i, j, k) : i <= 20,    j <= 20, k <= 10000
% train_lbl (i, j)    : i <= 60000, j <= 1
% test_lbl  (i, j)    : i <= 10000, j <= 1
train_img_file  = 'data/MNIST/train-images.idx3-ubyte';
train_lbl_file  = 'data/MNIST/train-labels-idx1-ubyte';
test_img_file   = 'data/MNIST/t10k-images-idx3-ubyte';
test_lbl_file   = 'data/MNIST/t10k-labels-idx1-ubyte';
% read data sets
[train_img, train_lbl] = readMNIST( train_img_file, train_lbl_file, siz_load + siz_val, 0 ) ;
[test_img,  test_lbl]  = readMNIST( test_img_file,  test_lbl_file,  siz_test, 0 ) ;
% max pool with poolsize 2 and stride 2
% train_img       = sepblockfun(train_img, [2, 2], 'max' ) ;
% test_img        = sepblockfun(test_img, [2, 2], 'max' ) ;
% useful parameters
size_image      = size(train_img, 1) ;
num_neurons     = size_image^2 ;
% training set
img_train       = reshape( train_img(:, :, 1:siz_load), [num_neurons, siz_load] ) ;
lab_train       = train_lbl(1:siz_load) ; 
% validation set
img_val         = reshape( train_img(:, :, end-siz_val+1:end), [num_neurons, siz_val] ) ;
lab_val         = train_lbl(end-siz_val+1:end) ;
% test set
img_test        = reshape( test_img, [num_neurons, siz_test] ) ;
lab_test        = test_lbl ;

% remove unnecessary variables from workspace
clear train_img train_lbl test_img test_lbl

disp("data loaded")

%% Select images to memorize

% select random patterns
% if num_patterns == size(train_img, 3)
%     patterns = train_img ; 
%     labels   = train_lbl ;
% else
%     rand_idx = randi(60000, 1, num_patterns) ;
%     patterns = train_img(rand_idx) ;
%     labels   = train_lbl(rand_idx) ;
% end

% select means of each group
[patterns, labels] = select_means( img_train, lab_train, num_protos, 'rbf', 10 ) ;

% select principal components
% sig = 0.1 ;
% patterns = select_KPCA( train_img, train_lbl, num_protos, 'RBF', sig ) ;

% visualize chosen prototypes
% figure('position', [100, 100, 1000, num_protos*100])
% for i = 1:size(patterns, 3)
%   for j = 1:num_protos
%       subplot( num_protos, size(patterns, 3), (j-1)*size(patterns, 3) + i )
%       plot_image( squeeze(patterns(:, j, i)), size_image, " " )
%   end
% end
% sgtitle( num2str(sig) )

patterns = reshape( patterns, [num_neurons, num_protos*size(patterns, 3)] ) ;
disp("patterns selected")

%% compute L2 distance between selected patterns

dist = zeros(10, 10) ;
for k = 0:9
    for l = (k+1):9
        disp("compare " + num2str(labels(1+k*num_protos)) + "and" + num2str(labels(1+l*num_protos)))
        dist(k+1, l+1) = phiTphi( patterns(:, 1+k*num_protos), patterns(:, 1+l*num_protos), 'L2' ) ;
    end
end
dist

%% Build model to memorize patterns
model = Memory_Model( ) ;
% add layers
for l = 1:num_layers
    model = model.add_layer( formulation{l}, dim_layer{l}, ...
                             p_err{l}, p_drv{l}, p_reg{l}, ...
                             feature_map{l}, parameter{l} ) ;
end

% train model
model = model.train( patterns ) ;

% perf = 0 ; 
% for k = 1:k_cross_val
%     
%     % evaluate model on validation set
%     
%     % running average of clasiication performance
% end



%% test on new image
% select test image 
random_idx = randi(siz_test) ;
test_x     = img_test(:, random_idx) ;
test_y     = lab_test(random_idx) ;

path = model.simulate(test_x(:)) ;
num_steps = size(path, 3) ;

% store closest index (L1 distance)
dist = phiTphi(patterns, path(:, 1, end), 'L1') ;
[~,  closest_idx] = sort(dist) ;
found_label = mode( labels( closest_idx(1:num_to_avg_over) ) ) ;

% PLOT results
max_steps_show = 5 ;
num_plots = min(num_steps + 2, max_steps_show + 2);
% plot images as [test, step i, closest to]
figure('position', [200, 200, 800, 200])
% original image
subplot(1, num_plots, 1)
plot_image(test_x(:), size_image, "corrupted")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
    
for i = 1:num_plots-2
    name = "step " + num2str(i*max(1, floor((num_steps-1)/max_steps_show))) ;
    subplot(1, num_plots, i+1)
    plot_image( path(:, 1,  i*max(1, floor((num_steps-1)/max_steps_show))), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end

subplot(1, num_plots, num_plots)
plot_image( patterns(:, closest_idx(1)), size_image, "closest to")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")


%% test on corrupted image from training set

random_idx = randi(siz_train) ;
test_x     = patterns(:, random_idx) + 1*randn(400, 1) ;
test_y     = labels(random_idx) ;

path = model.simulate(test_x(:)) ;
num_steps = size(path, 3) ;

% store closest index (L1 distance)
dist = phiTphi(patterns, path(:, 1, end), 'L1') ;
[~,  closest_idx] = sort(dist) ;
found_label = mode( labels( closest_idx(1:num_to_avg_over) ) ) ;

% PLOT results
max_steps_show = 5 ;
num_plots = min(num_steps + 3, max_steps_show + 3);
% plot images as [test, step i, closest to]
figure('position', [200, 200, 800, 200])
% original image
subplot(1, num_plots, 1)
plot_image(patterns(:, random_idx), size_image, "original")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
% corrupted image
subplot(1, num_plots, 2)
plot_image(test_x(:), size_image, "corrupted")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
    
for i = 1:num_plots-3
    name = "step " + num2str(i*max(1, floor((num_steps-1)/max_steps_show))) ;
    subplot(1, num_plots, i+2)
    plot_image( path(:, 1,  i*max(1, floor((num_steps-1)/max_steps_show))), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end

subplot(1, num_plots, num_plots)
plot_image( patterns(:, closest_idx(1)), size_image, "closest to")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")

%% walk on manifold from one pattern to another

img_1 = patterns( :, randi(siz_train) ) ;
img_2 = patterns( :, randi(siz_train) ) ;

walk = model.walk_on_manifold( img_1, img_2, 1 ) ;
num_steps = size(walk, 2) ;

% PLOT results
max_steps_show = 10 ;
num_plots = min(num_steps + 2, max_steps_show + 2);
% plot images as [test, step i, closest to]
figure('position', [200, 200, 800, 200])
% start image
subplot(1, num_plots, 1)
plot_image(img_1, size_image, "start")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
% walk
for i = 1:num_plots-2
    name = "step " + num2str(i*max(1, floor((num_steps-1)/max_steps_show))) ;
    subplot(1, num_plots, i+1)
    plot_image( walk(:,  i*max(1, floor((num_steps-1)/max_steps_show))), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end
% end image
subplot(1, num_plots, num_plots)
plot_image(img_2, size_image, "end")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

%% visualize direction of largest singular vectors

num_sv_to_plot = 5 ;

% choose image
img = patterns( :, randi(siz_train) ) ;

[U, S] = model.jacobian_SVD( num_sv_to_plot, img ) ;

S

% PLOT results
figure('position', [200, 200, 800, 200])
% start image
subplot(1, num_sv_to_plot+1, 1)
plot_image(img, size_image, "state")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
% plot singular vectors
for i = 1:num_sv_to_plot
    % rescale vector to make it visible
    m = min(U(:, 1, i)) ;
    M = max(U(:, 1, i)) ;
    scale = min( abs(0.5/M), abs(-0.5/m) ) ;
    
    
    name = "vector " + num2str(i) ;
    subplot(1, num_sv_to_plot+1, i+1)
    plot_image( 0.5+ scale*U(:, 1, i), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end

% move point a little bit in direction of vector 2
% PLOT results
figure('position', [200, 200, 800, 200])
% start image
subplot(1, num_sv_to_plot+1, 1)
plot_image(img, size_image, "start")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
% plot singular vectors
for i = 1:num_sv_to_plot
    % rescale vector to make it visible
    dx = i*1 ;
    subplot(1, num_sv_to_plot+1, i+1)
    plot_image( img + dx*U(:, 1, 1), size_image, strcat("new point with Dx = ", num2str(dx)))
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end


