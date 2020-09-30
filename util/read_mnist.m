% Created  by OctaveOliviers
%          on 2020-09-27 13:04:01
%
% Modified on 2020-09-30 10:56:32

%% read_mnist: create MNIST dataset
function [varargout] = read_mnist(varargin)
    % varargin
    %       pool_size
    %       num_train
    %       num_test

    % varargout
    %       data_train
    %       data_test
    %       label_train
    %       label_test

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
    if nargin==2 && nargout==1
        % only load training data
        [train_img, ~] = readMNIST( train_img_file, train_lbl_file, varargin{2}, 0 ) ;
        train_img      = sepblockfun(train_img, [varargin{1}, varargin{1}], 'max' ) ;
        %
        num_neurons    = size(train_img, 1)^2 ;
        varargout{1}   = reshape( train_img(:, :, 1:varargin{2}), [num_neurons, varargin{2}] ) ;

    elseif nargin==3 && nargout==2
        % load training and testing data
        [train_img, ~] = readMNIST( train_img_file, train_lbl_file, varargin{2}, 0 ) ;
        [test_img,  ~] = readMNIST( test_img_file,  test_lbl_file,  varargin{3}, 0 ) ;
        train_img      = sepblockfun(train_img, [varargin{1}, varargin{1}], 'max' ) ;
        test_img       = sepblockfun(test_img, [varargin{1}, varargin{1}], 'max' ) ;
        %
        num_neurons    = size(train_img, 1)^2 ;
        varargout{1}   = reshape( train_img(:, :, 1:varargin{2}), [num_neurons, varargin{2}] ) ;
        varargout{2}   = reshape( test_img(:, :, 1:varargin{3}), [num_neurons, varargin{3}] ) ;

    elseif nargin==3 && nargout == 4
        % load training and testing data, as well as the labels of each
        [train_img, varargout{3}] = readMNIST( train_img_file, train_lbl_file, varargin{2}, 0 ) ;
        [test_img,  varargout{4}]  = readMNIST( test_img_file,  test_lbl_file,  varargin{3}, 0 ) ;
        train_img      = sepblockfun(train_img, [varargin{1}, varargin{1}], 'max' ) ;
        test_img       = sepblockfun(test_img, [varargin{1}, varargin{1}], 'max' ) ;
        %
        num_neurons    = size(train_img, 1)^2 ;
        varargout{1}   = reshape( train_img(:, :, 1:varargin{2}), [num_neurons, varargin{2}] ) ;
        varargout{2}   = reshape( test_img(:, :, 1:varargin{3}), [num_neurons, varargin{3}] ) ;

    else
        error("Not correct number of input and output arguments")
    end
end