% Created  by OctaveOliviers
%          on 2020-03-29 17:06:12
%
% Modified on 2020-06-04 13:10:45

% Experiment on MNIST data set

% cls_mnist("dual", 400, 1e2, 1e0, 1e-2, "rbf", 20)

function cls_mnist(formulation, dim_layer, p_err, p_drv, p_reg, feat_map, feat_param)

%     feature('SetPrecision', 24)
    rng(42) ;

    addpath( '../models/' )
    addpath( '../util/' )

    %% load data
    train_img = readmatrix('../../data/MNIST/train-images-50-original.csv')' ;
    % remove row and column numbers     
    train_img = train_img(2:end, 2:end) ;
    % switch from row-major to column-major
    [dim_img, num_img] = size(train_img) ;
    train_img = reshape(train_img, [sqrt(dim_img), sqrt(dim_img), num_img]) ;
    train_img = permute(train_img, [2, 1, 3]) ;
    train_img = reshape(train_img, [dim_img, num_img]) ;
    
    disp("Loaded data")

    %% Build model to store patterns
    model = CLSSVM( ) ;
    model = model.add_layer(formulation, dim_layer, ...
                            p_err, p_drv, p_reg, ...
                            feat_map, feat_param) ;
    model = model.train( train_img(:,1:40) ) ;

    disp("Trained model")

    %% test on corrupted image from training set

    % 41
    % 8, 2, 48, 18, 16
    test_idx = 41 ;
    test_img = train_img(:, test_idx) ;

    for i = 1:10^3
        test_img = model.simulate_one_step(test_img) ;
    end

    % PLOT results
    fig_pos = [200, 200, 200, 200] ;

    % original image
    figure('position', fig_pos)
    plot_image(train_img(:, test_idx))
    
    % retrieved image
    figure('position', fig_pos)
    plot_image(test_img)
    
end
