% Created  by OctaveOliviers
%          on 2020-03-29 17:06:12
%
% Modified on 2020-05-08 10:20:29

clear all
clc
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of data set
siz_train       = 100 ;
siz_val         = 100 ;
siz_test        = 100 ; 
num_protos      = 3 ;       % number of prototypes to select for each label
% parameters of model
num_layers      = 1 ;
formulation     = { 'dual' } ; 
dim_layer       = { 400 } ;
feature_map     = { 'rbf' } ; 
parameter       = { 1 } ;
p_err           = { 1e4 } ;  % importance of error
p_reg           = { 1e1 } ;  % importance of regularization
p_drv           = { 1e3 } ;  % importance of minimizing derivative
% parameters of training
k_cross_val     = 10 ;
% parameters of testing
tolerance       = 1e-3 ;
num_to_avg_over = 1 ;


% LOAD dataset
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
[train_img, train_lbl] = readMNIST( train_img_file, train_lbl_file, siz_train + siz_val, 0 ) ;
[test_img,  test_lbl]  = readMNIST( test_img_file,  test_lbl_file,  siz_test, 0 ) ;
% useful parameters
size_image      = size(train_img, 1) ;
num_neurons     = size_image^2 ;
% split data set in training and validation set
img_4_train     = reshape( train_img(:, :, 1:siz_train), [num_neurons, siz_train] ) ;
lab_4_train     = train_lbl(1:siz_train) ;
%
img_4_val       = reshape( train_img(:, :, end-siz_val+1:end), [num_neurons, siz_val] ) ;
lab_4_val       = train_lbl(end-siz_val+1:end) ;
%
img_4_test      = reshape( test_img, [num_neurons, siz_test] ) ;
lab_4_test      = test_lbl ;

% remove unnecessary variables from workspace
clear train_img train_lbl test_img test_lbl

disp("data loaded")


% SELECT patterns

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
% patterns = select_means( train_img, train_lbl, num_protos, 'rbf', 0.1 ) ;

% select principal components
% sig = 0.1 ;
% patterns = select_KPCA( train_img, train_lbl, num_protos, 'RBF', sig ) ;

% % visualize chosen prototypes
% figure('position', [100, 100, 1000, num_protos*100])
% for i = 1:size(patterns, 3)
%   for j = 1:num_protos
%       subplot( num_protos, size(patterns, 3), (j-1)*size(patterns, 3) + i )
%       plot_image( squeeze(patterns(:, j, i)), size_image, " " )
%   end
% end
% suptitle( num2str(sig) )

% patterns = reshape( patterns, [num_neurons, num_protos*size(patterns, 3)] ) ;
% disp("patterns selected")



%% Build model to memorize patterns
model = Memory_Model( ) ;
% add layers
for l = 1:num_layers

    model = model.add_layer( formulation{l}, dim_layer{l}, ...
                             p_err{l}, p_drv{l}, p_reg{l}, ...
                             feature_map{l}, parameter{l} ) ;

end


model = model.train( img_4_train ) ;


%% test on new image
% select test image 
random_idx = randi(siz_test) ;
test_x     = img_4_test(:, random_idx) ;
test_y     = lab_4_test(random_idx) ;

path = model.simulate(test_x(:)) ;

% store closest index (L1 distance)
dist = sum(abs(img_4_train - path(:, 1, end)), 1) ;
[~,  closest_idx] = sort(dist);
found_label = mode( lab_4_train( closest_idx(1:num_to_avg_over) ) ) ;

% PLOT results
num_plots = size(path, 2) + 2;
% plot images as [test, step i, closest to]
figure('position', [200, 200, 800, 200])
subplot(1, num_plots, 1)
plot_image(test_x(:), size_image, "initial")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
    
for i = 1:num_plots-2
    name = "step " + num2str(i) ;
    subplot(1, num_plots, i+1)
    plot_image(path(:, i), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end

subplot(1, num_plots, num_plots)
plot_image(img_4_train(:, closest_idx(1)), size_image, "closest to")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")