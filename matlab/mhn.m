
% test retrieval of patterns with Modern Hopfield Network

clear all
clc
rng(10) ;

% import dependencies
addpath( './models/' )
addpath( './util/' )

% parameters of data set
siz_train       = 50 ;
siz_load        = 50 ;
siz_val         = 100 ;
siz_test        = 1000 ; 


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
if siz_train == size(img_train, 2)
    patterns = img_train ; 
    labels   = lab_train ;
else
    rand_idx = randi(siz_load, 1, siz_train) ;
    patterns = img_train(:, rand_idx) ;
    labels   = lab_train(rand_idx) ;
end

% select means of each group
% [patterns, labels] = select_means( img_train, lab_train, num_protos, 'rbf', 0 ) ;

% select principal components
% sig = 0.1 ;
% patterns = select_KPCA( train_img, train_lbl, num_protos, 'RBF', sig ) ;

% patterns = reshape( patterns, [num_neurons, num_protos*size(patterns, 3)] ) ;

% % select number to store one specific digit
% selected = 0 ;
% idx = find( lab_train == selected ) ;
% patterns = img_train( :, idx(1:siz_train) ) ;

disp("patterns selected")

% visualize chosen patterns
% figure('position', [100, 100, 1000, num_protos*100])
% for i = 1:size(patterns, 2)
%   for j = 1:num_protos
%       subplot( num_protos, size(patterns, 2), (j-1)*size(patterns, 2) + i )
%       plot_image( patterns(:, i), size_image, " " )
%   end
% end
% sgtitle( num2str(sig) )


%% test on corrupted image from training set

random_idx = randi(siz_train) ;
% random_idx = 26 ;
test_x     = patterns(:, random_idx) + 1*randn(400, 1) ;

path = mhn_simulate(b, patterns, test_x) ;

% PLOT results
size_x = 150 ;
size_y = 150 ;

for p = 1:length(path)
    
    figure('position', [200, 200, size_x, size_y])
    plot_image( patterns(:, random_idx), size_image, "true")
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
    
    figure('position', [200, 200 + 200, size_x, size_y])
    plot_image( path{p}(:, 1), size_image, "original")
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;

    figure('position', [200, 200 + 400, size_x, size_y])
    plot_image( path{p}(:, end), size_image, "reconstruction")
    xlim([1 size_image]) ;
    ylim([1 size_image]) ;
end
    

%% Build model to memorize patterns

function path =  mhn_simulate(b, X, start)

    % parameters of model
    b = 1 ;
    % parameters of testing
    tol = 1e-1 ;

    [N,P] = size(start) ;

    max_steps = 50 ;

    % variable to store evolution of state
    path = cell(1,P) ;
    
    for p = 1:P
        
        path{p} = zeros(N,max_steps) ;
        path{p}(:,1) = start(:,p) ;
        
        x_old = start(:,p) ;
        
        for k = 2:max_steps
            x_new = X * softmax( b * X' * x_old ) ;
            path{p}(:,k) = x_new ;

            if norm(x_new-x_old) <= tol
                break
            end

            x_old = x_new ;
        end
        
        % remove all the zero columns in path
        path{p}(:, ~any(path{p},1)) = [] ;
    end
    
    
end
