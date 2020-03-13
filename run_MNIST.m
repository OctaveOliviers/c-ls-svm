% @Author: OctaveOliviers
% @Date:   2020-03-13 18:56:55
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-13 18:59:35

clear all
clc


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