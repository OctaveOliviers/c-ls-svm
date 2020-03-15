% @Author: OctaveOliviers
% @Date:   2020-03-13 18:49:42
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-13 18:55:39

clear all
clc

%% LOAD dataset

train_img_file = 'data/MNIST/train-images.idx3-ubyte';
train_lbl_file = 'data/MNIST/train-labels-idx1-ubyte';
test_img_file  = 'data/MNIST/t10k-images-idx3-ubyte';
test_lbl_file  = 'data/MNIST/t10k-labels-idx1-ubyte';

% each is a b&w image of 20 x 20 pixels
% train_img (i, j, k) : i <= 20,    j <= 20, k <= 60000
% test_img  (i, j, k) : i <= 20,    j <= 20, k <= 10000
% train_lbl (i, j)    : i <= 60000, j <= 1
% test_lbl  (i, j)    : i <= 10000, j <= 1
[train_img, train_lbl] = readMNIST(train_img_file, train_lbl_file, 60000, 0);
[test_img,  test_lbl]  = readMNIST(test_img_file,  test_lbl_file,  10000, 0);
size_image = size(train_img, 1);
num_neurons = size_image^2;

disp("data loaded")

%% PARAMETERS to play with

% patterns
num_patterns     = 60000;    % number of patterns to store in state space
% model
kernel_function  = 'softmax';
kernel_parameter = 0.5;    % bandwith of gaussian
formulation      = 'dual'; % solve in primal or dual formulation
num_layers       = 1;
% testing
num_steps        = 5;
tolerance        = 1e-3;
num_to_avg_over  = 1;

disp("new parameters")

%% SELECT num_patterns random patterns

if num_patterns == size(train_img, 3)
    patterns = reshape(train_img, [num_neurons, size(train_img, 3)]);
    labels   = train_lbl;
else
    patterns = zeros(num_neurons, num_patterns);
    labels   = zeros(1, num_patterns);
    for i = 1:num_patterns
        rand_idx = randi(60000);
        % store vectorized pattern
        patterns(:, i) = reshape(train_img(:, :, rand_idx), [num_neurons, 1]);
        % store label
        labels(:, i) = train_lbl(rand_idx);
    end
end

disp("patterns selected")

%% BUILD model

lambda = 1e6;
eta = 1e0 ; 

[hidden_units, bias] = build_model(patterns, formulation, 'rbf', 0.1, lambda, eta);


%% display kernel matrix

size_to_show = num_patterns;    %num_patterns

figure();
imagesc(A(end-size_to_show+1:end, end-size_to_show+1:end));
colorbar

%% dislay hidden units

h = 10;

figure();
subplot(1, 4, 1)
plot_image(patterns(:, h), size_image, 'pattern')
xlim([1 size_image]);
ylim([1 size_image]);
axis equal;

% subplot(1, 4, 2)
% plot_image(bias, size_image, 'bias')
% xlim([1 size_image]);
% ylim([1 size_image]);
% axis equal;
% 
% subplot(1, 4, 3)
% plot_image(hidden_units(:, h)+bias, size_image, 'hidden+bias')
% xlim([1 size_image]);
% ylim([1 size_image]);
% axis equal;

subplot(1, 4, 4)
plot_image(hidden_units(:, h), size_image, 'hidden vector')
xlim([1 size_image]);
ylim([1 size_image]);
axis equal;

%% TEST on new data

% select test pattern 
random_idx = randi(10000);
test_x     = test_img(:, :, random_idx);
test_y     = test_lbl(random_idx);

x    = test_x(:);
path = zeros(num_neurons, num_steps);
i    = 0;
diff = Inf;

while diff > tolerance && i < num_steps
    i = i+1;
    disp("iteration : " + num2str(i));
            
    % store previous state
    path(:, i) = x;
          
    % update state
    switch formulation
        case {'primal', 'p'}
            x = W' * f_map(x) + bias;
            
        case {'dual', 'd'}
            [weights, nrg] = kernel_weights(patterns, x, kernel_function, kernel_parameter);
            x = 1/eta * hidden_units * weights ; %+ bias;
    end
    disp("energy of state = " + num2str(-nrg))
    
    % store closest index (L1 distance)
    dist = sum(abs(patterns - x), 1);
    [~,  closest_idx] = sort(dist);
    found_label = mode( labels( closest_idx(1:num_to_avg_over) ) );
          
    % check for convergence
    diff = sum( abs( x-path(:, i) ) );
            
end

path = path(:, 1:i);

disp("original image was           : " + num2str(test_y))
disp("final state closest to image : " + num2str(found_label))
disp(" ")


% PLOT results
%plot_image(train_img(:, :, 5), 20, 'nothing')
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



%% VALIDATION PERFORMANCE

num_test  = length(test_lbl);
num_wrong = 0;

for i = 1:num_test
    i
    x    = reshape(test_img(:, :, i), [num_neurons, 1]);
    step = 0;
    diff = Inf;

    while diff > tolerance && step < num_steps
        step = step+1;

        % store previous state
        x_prev = x;

        % update state
%         switch formulation
%             case {'primal', 'p'}
%                 x = W' * f_map(x) + bias;
% 
%             case {'dual', 'd'}
                [weights, nrg] = kernel_weights(patterns, x, kernel_function, kernel_parameter);
                x = 1/eta * hidden_units * weights ; %+ bias;
%         end

        % store closest index (L1 distance)
        dist = sum(abs(patterns - x), 1);
        [~,  closest_idx] = sort(dist);
        found_label = mode( labels( closest_idx(1:num_to_avg_over) ) );

        % check for convergence
        diff = sum( abs( x-x_prev ) );
    end
    
    num_wrong = num_wrong + (found_label ~= test_lbl(i));
end

disp("Performance on test set = " + num2str(100*num_wrong/num_test) + " %");
