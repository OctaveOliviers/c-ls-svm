% Created  by OctaveOliviers
%          on 2020-03-29 19:34:42
%
% Modified on 2020-03-29 19:34:55

% from https://github.com/brendenlake/omniglot/tree/master/matlab

function plot_image(img, size_image, name)
    img = reshape(img, [size_image, size_image]);
    
    new_img = img;
    %new_img(isinf(img)) = 0.5;
    
    image([1 size_image],[1 size_image],repmat(new_img,[1 1 3]));
    set(gca,'YDir','reverse','XTick',[],'YTick',[]);
    xlim([1 size_image]);
    ylim([1 size_image]);
    axis equal;
    title(name)
end