% Created  by OctaveOliviers
%          on 2020-03-28 11:22:11
%
% Modified on 2020-05-12 09:33:52

% select the k data points closest to the mean of each cluster
%
%   input
%       data        data points in columns          matrix( dim_data x num_data )
%       labels      labels of each data point       matrix( 1 x num_data ) of integers
%       k           how many data points to select  integer
%       d_measure   distance measure, e.g. euclidean (L2), L1, RBF
%       varargin    contains parameter of distance measure if necessary
%
%   output
%       protos      matrix of selected prototypes   matrix( dim_data x k x num_clusters )

function [protos, sorted_labels] = select_means(data, labels, k, d_measure, varargin)

    % check correctness of input
    assert( size(data, 2)==length(labels), ...
            'Number of data points does not match number of labels.' )

    % extract useful information
    [dim_data, num_data]    = size(data) ;
    uni_labels  = unique(labels) ;
    num_labels  = length(uni_labels) ;

    % preallocate memory
    protos  = zeros( dim_data, k, num_labels ) ;
    sorted_labels = zeros( 1, k*num_labels ) ;

    for i = 1:num_labels

        % select data from l^th cluster
        data_l = data( :, labels == uni_labels(i) ) ;

        %
        sorted_labels((i-1)*k+1:i*k) = uni_labels(i) ;

        % mean of X
        mean_l = mean(data_l, 2) ;

        % compute similarity between mean and each cluster element
        D = phiTphi( mean_l, data_l, d_measure, varargin{:} ) ;
        % disp("label "+ num2str( uni_labels(i) ) )
        % disp("  min "+ num2str( min(D)) )
        % disp("  max "+ num2str( max(D)) )

        % fill prototypes
        [~, order]      = sort(D) ;
        sorted_X        = data_l(:, order) ;

        % choose prototypes
        for j = 1:min(k, size(data_l, 2))
            protos(:,j, i) = sorted_X(:, j) ;
        end
        % if did not have k data points in cluster l, copy some data points
        if j < k
            to_fill = k-j 
            num_copy = floor(to_fill/j) ;
            protos(:, j+1:j+num_copy*j, i) = repmat(protos(:, 1:j, i), [1, num_copy]) ;
            %
            still_to_fill = k - (j + num_copy*j) ;
            protos(:, end-still_to_fill+1:end, i) = protos(:, 1:still_to_fill, i) ;
        end

    end

end