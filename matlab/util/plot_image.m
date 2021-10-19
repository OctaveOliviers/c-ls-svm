% Created  by OctaveOliviers
%          on 2020-03-29 19:34:42
%
% Modified on 2020-05-12 09:55:38

% from https://github.com/brendenlake/omniglot/tree/master/matlab

function plot_image(img)
    size_img = sqrt(numel(img)) ;
    img = reshape(img, [size_img, size_img]);
    
    %new_img = img;
    %new_img(isinf(img)) = 0.5;
    
    image([1 size_img],[1 size_img],repmat(img,[1 1 3]));
    set(gca,'YDir','reverse','XTick',[],'YTick',[]);
    xlim([1 size_img]);
    ylim([1 size_img]);
    axis equal;
    % title(title, 'interpreter', 'latex', 'fontsize', 14)
end