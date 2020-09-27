% Created  by OctaveOliviers
%          on 2020-09-24 16:55:15
%
% Modified on 2020-09-26 15:36:40

clear all
clc

% set random number generator
rng(10)

% import dependencies
addpath( './models/' )
addpath( './util/' )

% load data
small = true ;  % load small dataset
num_img = 10 ;  % number of images to load
pool_size = 5 ; % size of pooling (should be divider of 15)
img = read_omniglot(small, num_img, pool_size) ;
siz_img = size(img, 1) ;

% noise one of the images
ratio = 0.2 ;
noise = [ -ones(ceil(ratio*siz_img), 1) ; ones(siz_img-ceil(ratio*siz_img), 1) ] ;
noise = noise( randperm(siz_img) ) ;
rnd_img = randi(num_img) ;
img_noise = img(:, rnd_img) .* noise ;

% visualize selected image
figure 
plot_image(img(:, rnd_img), sqrt(siz_img), 'original')

% create Dense Associative Memory
model = Dense_Associative_Memory( img, 'repol', 4 ) ;

state = img_noise ;
figure
for i = 1:500
    i

    state = model.simulate_one_step(state) ;

    plot_image(state, sqrt(siz_img), 'reconstructed')
    pause(0.05)
end
