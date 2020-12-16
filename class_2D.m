% Created  by OctaveOliviers
%          on 2020-10-14 10:28:07
%
% Modified on 2020-10-14 12:00:37

clear all
rng(10)

% add the folders to the Matlab path
addpath( './models/' )
addpath( './util/' )

% create data set
dim = 2 ;
num = 50 ;
%
mu_1 = [ 6 ; 0 ] ;
std_1 = 2 ;
class_1 = mu_1 + std_1 * randn( dim, num ) ;
%
mu_2 = [ 0 ; 6 ] ;
std_2 = 1 ;
class_2 = mu_2 + std_2 * randn( dim, num ) ;
%
mu_3 = [ 1 ; 3 ] ;
std_3 = 1.5 ;
class_3 = mu_3 + std_3 * randn( dim, num ) ;
%
% lab = [ ones(1, size(class_1, 2)) ; -ones(1, size(class_2, 2)) ] ;


% (hyper-)parameters of the layer
space           = 'dual' ;          % space to train layer
hp_equi         = 1e2 ;             % importance of equilibrium objective
hp_stab         = 1e1 ;             % importance of local stability objective
hp_reg          = 1e-2 ;            % importance of regularization
feat_map        = 'rbf' ;           % chosen feature map or kernel function
feat_map_param  = 8 ;               % parameter of feature map or kernel function
% build model
model = CLSSVM() ;
% add a layer
model = model.add_layer( space, dim, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;

% train model
model = model.train( [ class_1 , class_2 ] ) ;

% visualize trained model
% model.visualize( ) ;
plot_decision(model, class_1, class_2, mu_1, mu_2) ;


function plot_decision(model, class_1, class_2, mu_1, mu_2)

    figure('position', [100, 100, 400, 300])
    set(gca,'TickLabelInterpreter','latex')
    hold on
    box on
    
    % colors
    orange      = [230, 135, 28]/255 ;
    KUL_blue    = [0.11,0.55,0.69] ;
    green       = [58, 148, 22]/255 ;
    red         = [194, 52, 52]/255 ;
    grey        = 0.5 * [1 1 1] ;
    
    % parameters of the plot
    prec = 0.5 ;
    x_min = -3 + floor( min( [ class_1(1,:), class_2(1,:) ] )) ;
    x_max =  3 + ceil(  max( [ class_1(1,:), class_2(1,:) ] )) ;
    y_min = -3 + floor( min( [ class_1(2,:), class_2(2,:) ] )) ;
    y_max =  3 + ceil(  max( [ class_1(2,:), class_2(2,:) ] )) ;
    x = x_min:prec:x_max ;
    y = y_min:prec:y_max ;
    [X, Y] = meshgrid(x, y) ;           
    %
    
    data = [ X(:)' ; Y(:)' ] ;
    
%     [paths, ~, x_ends] = model.simulate( data ) ;
    
%     bin_class_1 = vecnorm(x_ends-mean(class_1, 2)) <= vecnorm(x_ends-mean(class_2, 2)) ;
%     bin_class_2 = vecnorm(x_ends-mean(class_2, 2)) <= vecnorm(x_ends-mean(class_2, 2)) ;
    
    for p = 1:size(data,2)
        [~, ~, x_end] = model.simulate( data(:,p) ) ;
        
        bin = vecnorm(x_end-mean(class_1, 2)) <= vecnorm(x_end-mean(class_2, 2)) ;
        
        scatter(data(1,p), data(2,p), 10, 'filled', 'MarkerFaceColor', orange + (1-bin)*(green-orange), 'MarkerFaceAlpha', .5)
    end
    
    % plot stream lines
    F = model.simulate_one_step( data ) ;
    f1 = reshape( F(1, :), [length(y), length(x)] ) ;
    f2 = reshape( F(2, :), [length(y), length(x)] ) ;
    %
    hlines = streamslice( X, Y, (f1-X), (f2-Y), 0.5) ;
    set(hlines,'LineWidth', 1, 'Color', grey)
    
    % class 1 patterns to memorize
    l_patterns_c1 = plot(class_1(1, :), class_1(2, :), 'x', 'MarkerSize', 10, 'color', orange, 'LineWidth', 2) ;
    plot(mu_1(1, :), mu_1(2, :), '.', 'MarkerSize', 20, 'color', red, 'LineWidth', 2) ;
    % class 2 patterns to memorize
    l_patterns_c2 = plot(class_2(1, :), class_2(2, :), '+', 'MarkerSize', 10, 'color', green, 'LineWidth', 2) ;
    plot(mu_2(1, :), mu_2(2, :), '.', 'MarkerSize', 20, 'color', red, 'LineWidth', 2) ;

    hold off
    set(gca,'FontSize',12)
    xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
    ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)
    xlim([x_min, x_max])
    ylim([y_min, y_max])
%     xticks([])
%     yticks([])
    axis equal
    title( "2D classification", 'interpreter', 'latex', 'fontsize', 14 )
    legend( [l_patterns_c1, l_patterns_c2, hlines(1)], {'Class 1', 'Class 2', 'Streamlines'}, 'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;
    % title( "Vector field learned by contractive autoencoder", 'interpreter', 'latex', 'fontsize', 14 )
    % legend( [l_patterns, l_man, qui, l_J_2], ...
    %         {'Data point $\mathbf{x}_p$', 'Manifold', 'Vectorfield', "Principal direction of jacobian"}, ...
    %         'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;

    % plot histogram of singular values
    % plot_singular_values( S ) ;

end