% Created  by OctaveOliviers
%          on 2020-03-29 17:06:12
%
% Modified on 2020-10-11 11:59:34

% Experiment classification on MNIST data set

clear all
clc
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of data set
siz_train       = 60 ;              % size of training set
siz_load        = 100 ;             % number of 
siz_val         = 100 ;             % size of validation set
siz_test        = 1000 ;            % size of test set
num_protos      = floor(siz_train/10) ;       % number of prototypes to select for each label
% parameters of model
num_layers      = 1 ;
formulation     = { 'dual', 'primal' } ; 
dim_layer       = { 400 , 400 } ;
feature_map     = { 'rbf', 'tanh' } ; 
parameter       = { 20, 30 } ;
p_err           = { 1e2,  1e2 } ;  % importance of error
p_drv           = { 1e2,  1e2 } ;  % importance of minimizing derivative
p_reg           = { 1e-2, 1e-2 } ;  % importance of regularization200
% parameters of training
% k_cross_val     = 10 ;
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

rand_idx = randi(siz_load, [1, siz_train]) ;
patterns = img_train(:, rand_idx) ;
labels   = lab_train(rand_idx) ;

% select means of each group
% [patterns, labels] = select_means( img_train, lab_train, num_protos, 'rbf', 10 ) ;

% visualize chosen prototypes
% figure('position', [100, 100, 1000, num_protos*100])
% for i = 1:size(patterns, 3)
%   for j = 1:num_protos
%       subplot( num_protos, size(patterns, 3), (j-1)*size(patterns, 3) + i )
%       plot_image( squeeze(patterns(:, j, i)), size_image, " " )
%   end
% end
% sgtitle( num2str(sig) )

% patterns = reshape( patterns, [num_neurons, num_protos*size(patterns, 3)] ) ;

disp("patterns selected")

%% compute L2 distance between selected patterns
% dist = zeros(10, 10) ;
% for k = 0:9
%     for l = (k+1):9
%         disp("compare " + num2str(labels(1+k*num_protos)) + "and" + num2str(labels(1+l*num_protos)))
%         dist(k+1, l+1) = phiTphi( patterns(:, 1+k*num_protos), patterns(:, 1+l*num_protos), 'L2' ) ;
%     end
% end
% dist


%% Build model to memorize patterns
model = CLSSVM( ) ;
% add layers
for l = 1:num_layers
    model = model.add_layer( formulation{l}, dim_layer{l}, ...
                             p_err{l}, p_drv{l}, p_reg{l}, ...
                             feature_map{l}, parameter{l} ) ;
end

model = model.train( patterns ) ;

%% Validate the parameters on the validation set
num_err = 0 ;

for i = 1:siz_val
    
    label = classify(model, img_val(:, i), img_train, lab_train) ;

    if ~(label == lab_val(i))
        num_err = num_err + 1 ;
    end
end

miss_class_perc = num_err / siz_val



%% test on new image
% select test image 
random_idx = randi(siz_test) ;
% random_idx = 32 ;
test_x     = img_test(:, random_idx) ;
test_y     = lab_test(random_idx) ;

label = classify(model, test_x, img_train, lab_train) ;

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(label))
disp(" ")


%% test on corrupted image from training set

random_idx = randi(siz_train) ;
% random_idx = 26 ;
test_x     = patterns(:, random_idx) + 1*randn(400, 1) ;
test_y     = labels(random_idx) ;

path = model.simulate(test_x(:)) ;
num_steps = size(path, 3) ;

% store closest index (L1 distance)
dist = phiTphi(patterns, path(:, 1, end), 'L1') ;
[~,  closest_idx] = sort(dist) ;
% found_label = mode( labels( closest_idx(1:num_to_avg_over) ) ) ;

size_x = 150 ;
size_y = 150 ;

% PLOT results
max_steps_show = 3 ;
num_plots = min(num_steps + 3, max_steps_show + 3);
% plot images as [test, step i, closest to]
figure('position', [200, 200, size_x, size_y])
% set(gcf,'renderer','Painters')
% set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')

% original image
% subplot(1, num_plots, 1)
plot_image(patterns(:, random_idx), size_image, "original")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

% corrupted image
% subplot(1, num_plots, 2)
figure('position', [200, 200, size_x, size_y])
plot_image(test_x(:), size_image, "corrupted")
xlim([1 size_image]) ;
ylim([1 size_image]) ;

% first step
figure('position', [200, 200, size_x, size_y])
plot_image( path(:, 1, 2), size_image, "step 1")

for i = 1:num_plots-3
    figure('position', [200, 200, size_x, size_y])
    name = "step " + num2str(i*max(1, floor((num_steps-1)/max_steps_show))) 
    subplot(1, num_plots, i+2)
    plot_image( path(:, 1,  i*max(1, floor((num_steps-1)/max_steps_show))), size_image, name)
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
    pos = get(gca, 'Position');
    pos(1) = 0.055;
    pos(3) = 0.9;
    set(gca, 'Position', pos)
end

figure('position', [200, 200, size_x, size_y])
subplot(1, num_plots, num_plots)
plot_image( path(:, 1, end), size_image, "reconstruction")
xlim([1 size_image]) ;
ylim([1 size_image]) ;
pos = get(gca, 'Position');
pos(1) = 0.055;
pos(3) = 0.9;
set(gca, 'Position', pos)
    
disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")




function label = classify(model, img, patterns, labels)

    [~, ~, x_recon] = model.simulate(img) ;
    dist    = vecnorm(patterns - x_recon, 2, 1) ;
    label   = labels( dist == min(dist) ) ;
    
end