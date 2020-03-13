% @Author: OctaveOliviers
% @Date:   2020-03-13 18:56:55
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-13 19:47:10

clear all
clc

% parameters to play with
% patterns
num_patterns    = 10 ;		% number of patterns to store in state space
% model
formulation 	= 'dual' ; 
feature_map 	= 'g' ; 
parameter 		= 1.5 ;
num_layers      = 1 ;
% testing
tolerance       = 1e-3 ;
num_to_avg_over	= 1 ;


% LOAD dataset
% each is a b&w image of 20 x 20 pixels
% train_img (i, j, k) : i <= 20,    j <= 20, k <= 60000
% test_img  (i, j, k) : i <= 20,    j <= 20, k <= 10000
% train_lbl (i, j)    : i <= 60000, j <= 1
% test_lbl  (i, j)    : i <= 10000, j <= 1
train_img_file = 'data/MNIST/train-images.idx3-ubyte';
train_lbl_file = 'data/MNIST/train-labels-idx1-ubyte';
test_img_file  = 'data/MNIST/t10k-images-idx3-ubyte';
test_lbl_file  = 'data/MNIST/t10k-labels-idx1-ubyte';
[train_img, train_lbl] = readMNIST(train_img_file, train_lbl_file, 60000, 0);
[test_img,  test_lbl]  = readMNIST(test_img_file,  test_lbl_file,  10000, 0);
size_image  = size(train_img, 1);
num_neurons = size_image^2;

disp("data loaded")


% SELECT patterns
if num_patterns == size(train_img, 3)
    patterns = reshape(train_img, [num_neurons, size(train_img, 3)]) 
    labels   = train_lbl ;
else
	rand_idx = randi(60000, 1, num_patterns) ;
    patterns = reshape( train_img(:, :, rand_idx), [num_neurons, num_patterns] ) ;
    labels   = train_lbl(rand_idx) ;
end

disp("patterns selected")


% Build model to memorize patterns
p_err  = 1e4 ;	% importance of error
p_reg  = 1e1 ;	% importance of regularization
p_drv  = 1e5 ;	% importance of minimizing derivative


model = Memory_Model_Shallow(formulation, feature_map, parameter, p_err, p_drv, p_reg) ;

model = model.train(patterns) ;


% test on new 
% select test pattern 
random_idx = randi(10000) ;
test_x     = test_img(:, :, random_idx) ;
test_y     = test_lbl(random_idx) ;


path = model.simulate(test_x(:)) ;

% store closest index (L1 distance)
dist = sum(abs(patterns - path(:, 1, end)), 1);
[~,  closest_idx] = sort(dist);
% found_label = mode( labels( closest_idx(1:num_to_avg_over) ) );

% PLOT results
num_plots = size(path, 2) + 2;
% plot images as [test, step i, closest to]
figure('position', [200, 200, 800, 200])
subplot(1, num_plots, 1)
plot_image(test_x(:), size_image, "initial")
xlim([1 size_image]);
ylim([1 size_image]);
    
for i = 1:num_plots-2
    name = "step " + num2str(i);
    subplot(1, num_plots, i+1)
    plot_image(path(:, i), size_image, name)
    xlim([1 size_image]);
    ylim([1 size_image]);
end

subplot(1, num_plots, num_plots)
plot_image(patterns(:, closest_idx(1)), size_image, "closest to")
xlim([1 size_image]);
ylim([1 size_image]);

% disp("original image was           : " + num2str(test_y))
% disp("final state closest to image : " + num2str(found_label))
% disp(" ")
