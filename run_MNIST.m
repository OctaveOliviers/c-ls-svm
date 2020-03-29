% @Author: OctaveOliviers
% @Date:   2020-03-13 18:56:55
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-21 15:53:57

% clear all
% clc

% % import dependencies
% addpath( './models/' )
% addpath( './support/' )

% parameters to play with
% patterns
num_protos      = 3 ;       % number of prototypes to select for each label
% num_patterns    = 10 ;        % number of patterns to store in state space
% model
formulation     = 'dual' ; 
feature_map     = 'g' ; 
parameter       = 1 ;
num_layers      = 1 ;
% testing
tolerance       = 1e-3 ;
num_to_avg_over = 1 ;


% % LOAD dataset
% % each is a b&w image of 20 x 20 pixels
% % train_img (i, j, k) : i <= 20,    j <= 20, k <= 60000
% % test_img  (i, j, k) : i <= 20,    j <= 20, k <= 10000
% % train_lbl (i, j)    : i <= 60000, j <= 1
% % test_lbl  (i, j)    : i <= 10000, j <= 1
% train_img_file  = 'data/MNIST/train-images.idx3-ubyte';
% train_lbl_file  = 'data/MNIST/train-labels-idx1-ubyte';
% test_img_file   = 'data/MNIST/t10k-images-idx3-ubyte';
% test_lbl_file   = 'data/MNIST/t10k-labels-idx1-ubyte';
% % read data sets
% [train_img, train_lbl] = readMNIST( train_img_file, train_lbl_file, 60000, 0 );
% [test_img,  test_lbl]  = readMNIST( test_img_file,  test_lbl_file,  10000, 0 );
% % useful parameters
% size_image      = size(train_img, 1);
% num_neurons     = size_image^2;
% % vectorize images
% train_img       = reshape( train_img, [num_neurons, size(train_img, 3)] ) ;
% test_img        = reshape( test_img, [num_neurons, size(test_img, 3)] ) ;

% disp("data loaded")


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
patterns = select_means( train_img, train_lbl, num_protos, 'rbf', 0.1 ) ;

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

patterns = reshape( patterns, [num_neurons, num_protos*size(patterns, 3)] ) ;
disp("patterns selected")



% % Build model to memorize patterns
p_err  = 1e4 ;  % importance of error
p_reg  = 1e1 ;  % importance of regularization
p_drv  = 1e3 ;  % importance of minimizing derivative


model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg ) ;

model = model.train( patterns ) ;


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

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")