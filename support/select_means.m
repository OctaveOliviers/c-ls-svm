% @Author: OctaveOliviers
% @Date:   2020-03-20 09:58:27
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-20 11:00:53

% select the k data points closest to the mean of each cluster
%
%	input
%		data		data points in columns			matrix( dim_data x num_data )
%		labels		labels of each data point 		matrix( 1 x num_data )
%		k 			how many data points to select	integer
%		d_measure	distance measure, e.g. euclidean (L2), L1, RBF
%		varargin 	contains parameter of distance measure if necessary
%
%	output
%		protos 		matrix of selected prototypes 	matrix( dim_data x k x num_clusters )

function protos = select_means(data, labels, k, d_measure, varargin)

	% check correctness of input
	assert( size(data, 2)==length(labels), ...
			'Number of data points and number of labels do not match' )

	% extract useful information
	[dim_data, num_data] 	= size(data) ;
	Labels		= unique(labels) ;
	num_labels 	= length(Labels) ;

	% preallocate memory
	protos 	= zeros( dim_data, k, num_labels ) ;

	for i = 1:num_labels

		% select data from l^th cluster
		data_l = data( :, labels == Labels(i) ) ;

		% mean of X
		mean_l = mean(data_l, 2) ;

		% compute similarity between mean and each cluster element
		D = phiTphi( mean_l, data_l, d_measure, varargin{:} ) ;

		% fill prototypes
		[~, order] 		= sort(D) ;
		sorted_X 		= data_l(:, order) ;
		protos(:, :, i) = sorted_X(:, 1:k) ;
	end

end