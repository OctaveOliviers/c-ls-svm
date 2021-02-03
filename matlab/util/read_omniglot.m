% Created  by OctaveOliviers
%          on 2020-09-26 13:30:29
%
% Modified on 2020-09-26 15:34:29

%% read_omniglot: create omniglot dataset
function data = read_omniglot(small, num_img, pool_size)
    
    if small
        load('data/Omniglot/data_background_small1.mat')
    else
        load('data/Omniglot/data_background.mat')
    end

    img = vertcat(images{1:end}) ;
    img = vertcat(img{1:end}) ;

    % select random images
    idx_sel = randi(length(img), num_img, 1) ;
    img_sel = img( idx_sel ) ;

    % process images: binarize, pool and vectorize
    img_bin = 2 * cat(3, img_sel{:}) - 1 ;
    img_small = sepblockfun(img_bin, [pool_size, pool_size], 'min' ) ;
    data = reshape( img_small, [ numel(img_small(:, :, 1)) , num_img] ) ;

end
