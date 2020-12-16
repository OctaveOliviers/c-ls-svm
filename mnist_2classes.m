function out = expe_mnist(sz_train, sz_val, sz_test, p_err, p_drv, p_reg, param)
% Created  by OctaveOliviers
%          on 2020-03-29 17:06:12
%
% Modified on 2020-10-11 11:59:34
% Experiment classification on MNIST data set
% Modified & extended by Henri De Plaen


%% PRELIMINARIES
% DATA SIZE PARAMETERS
size_train       = min(60000, sz_train) ;         % size of training set
size_val         = min(60000-sz_train, sz_val) ;  % size of validation set
size_test        = min(10000, sz_test) ;          % size of test set

% MODEL PARAMETERS
num_layers      = 1 ;
formulation     = { 'dual' } ;
dim_layer       = { 400 } ;
feature_map     = { 'rbf' } ;
parameter       = { param } ;
p_err           = { p_err } ;   % importance of error
p_drv           = { p_drv } ;   % importance of minimizing derivative
p_reg           = { p_reg } ;   % importance of regularization
% validation for later

% DISP
disp(' ') ;
disp('%%%%%%%%% MNIST EXPERIMENT %%%%%%%%%%%%%%%') ;
disp(['Training size: ' num2str(size_train)]) ;
disp(['Validation size: ' num2str(size_val)]) ;
disp(['Test size: ' num2str(size_test)]) ;
disp(' ');
disp('ERR       DRV       REG       ') ;
disp([num2str(p_err{1},'%-8.2E') '  ' num2str(p_drv{1},'%-8.2E') '  ' num2str(p_reg{1},'%-8.2E')]) ;
disp(' ') ;


%% LOAD & PREPROCESS DATA

% select digits
d1 = 1 ;
d2 = 2 ;

% LOAD
train_img_file  = './data/MNIST/train-images.idx3-ubyte';  % training images path
train_lbl_file  = './data/MNIST/train-labels-idx1-ubyte';  % training labels path
test_img_file   = './data/MNIST/t10k-images-idx3-ubyte';   % testing images path
test_lbl_file   = './data/MNIST/t10k-labels-idx1-ubyte';   % testing labels path

[train_images_all, train_labels_all] = readMNIST( ...
    train_img_file, train_lbl_file, 60000, 0 ) ;
[test_images_all,  test_labels_all]  = readMNIST( ...
    test_img_file,  test_lbl_file,  10000, 0 ) ;

disp("(1/4) Data loaded") ;

% DATA SELECTION
d1_tr_idx = 
d2_te_idx = 

num_all_tr = size(train_images_all, 3) ;    % number of datapoints in training dataset
num_all_te = size(test_images_all, 3) ;     % number of datapoints in testing dataset
idx_tr_all = randperm(num_all_tr) ;         % random permutation of training indices
idx_te_all = randperm(num_all_te) ;         % random permutation of testing indices

idx_tr  = idx_tr_all(1:size_train) ;                        % selected training indices in full training dataset
idx_val = idx_tr_all(size_train+1:size_train+size_val) ;    % selected validation indices in full training dataset
idx_te  = idx_te_all(1:size_test) ;                         % selected testing indices in full testing dataset

% PREPROCESS
num_neurons  = size(train_images_all, 1) * size(train_images_all, 2) ; % number of pixels in each image

train_images = reshape(train_images_all(:,:,idx_tr), [num_neurons, size_train]) ;
train_labels = train_labels_all(idx_tr) ;
val_images = reshape(train_images_all(:,:,idx_val), [num_neurons, size_val]) ;
val_labels = train_labels_all(idx_val) ;
test_images  = reshape(train_images_all(:,:,idx_te), [num_neurons, size_test]) ;
test_labels  = test_labels_all(idx_te) ;

% WORKSPACE LIGHTENING
clear train_images_all train_labels_all test_images_all test_labels_all
disp("(2/4) Preprocessing done") ;


%% INITILAZING & TRAINING MODEL
model = CLSSVM( ) ;

% ADD LAYERS
for l = 1:num_layers
    model = model.add_layer( formulation{l}, dim_layer{l}, ...
        p_err{l}, p_drv{l}, p_reg{l}, ...
        feature_map{l}, parameter{l} ) ;
end

% TRAINING
model = model.train(train_images) ;

disp('(3/4) Model trained') ;

if size_val~=0
    %% VALIDATING MODEL
    pred_labels(size_val) = -1 ;
    
    for i = 1:size_val
        loc_label = classify(model, val_images(:, i), train_images, train_labels) ;
        pred_labels(i) = loc_label ;
    end
    
    [conf_matrix, results] = confusion.getMatrix(val_labels(:), pred_labels(:),0) ;
    out.c_matrix = conf_matrix ;
    out.results = results ;
    
    disp('(4/4) Model validated') ;
    
    acc = results.Accuracy ;
    disp('') ;
    disp(['Accuracy: ' num2str(100*acc) '%']) ;
    disp('') ;
    
else
    %% TESTING MODEL
    pred_labels(size_test) = -1 ;
    
    for i = 1:size_test
        loc_label = classify(model, test_images(:, i), train_images, train_labels) ;
        pred_labels(i) = loc_label ;
    end
    
    [conf_matrix, results] = confusion.getMatrix(test_labels(:), pred_labels(:),0) ;
    out.c_matrix = conf_matrix ;
    out.results = results ;
    
    disp('(4/4) Model tested') ;
    
    acc = results.Accuracy ;
    disp(' ') ;
    disp(['Accuracy: ' num2str(100*acc) '%']) ;
    disp(' ') ;
end

%% OUT
out.hyperparameters.p_err = p_err ;
out.hyperparameters.p_drv = p_drv ;
out.hyperparameters.p_reg = p_reg ;
out.hyperparameters.sigma = parameter ;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function label = classify(model, img, patterns, labels)

[~, ~, x_recon] = model.simulate(img) ;
dist            = vecnorm(patterns - x_recon, 2, 1) ;
[~, idx]        = min(dist) ;
label           = labels(idx) ;

end
