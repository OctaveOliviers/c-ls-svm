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
num = 20 ;
%
mu_1 = [ -3 ; 0 ] ;
std_1 = 1.5 ;
class_1 = mu_1 + std_1 * randn( dim, num ) ;
%
mu_2 = [ 3 ; 0 ] ;
std_2 = 1 ;
class_2 = mu_2 + std_2 * randn( dim, num ) ;
%
mu_3 = [ 0 ; 5.2 ] ;
std_3 = 0.5 ;
class_3 = mu_3 + std_3 * randn( dim, num ) ;
%
% lab = [ ones(1, size(class_1, 2)) ; -ones(1, size(class_2, 2)) ] ;


% (hyper-)parameters of the layer
space           = 'dual' ;          % space to train layer
hp_equi         = 1e0 ;             % importance of equilibrium objective
hp_stab         = 1e0 ;             % importance of local stability objective
hp_reg          = 1e0 ;            % importance of regularization
feat_map        = 'rbf' ;           % chosen feature map or kernel function
feat_map_param  = 4 ;               % parameter of feature map or kernel function
% build model
model = CLSSVM() ;
% add a layer
model = model.add_layer( space, dim, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;

% train model
%model = model.train( [ class_1 , class_2 ] ) ;
%model = model.train( [ class_1 , class_2 , class_3 ] ) ;
% train model only on mean of each class
%model = model.train( [ mu_1 , mu_2 ] ) ;
model = model.train( [ mu_1 , mu_2, mu_3 ] ) ;

% visualize trained model
% model.visualize( ) ;
plot_decision(model, class_1, class_2, class_3, mu_1, mu_2, mu_3) ;
plot_convergence(model, class_1, class_2, mu_1, mu_2) ;


%function plot_decision(model, class_1, class_2, mu_1, mu_2)
function plot_decision(model, class_1, class_2, class_3, mu_1, mu_2, mu_3)

    figure('position', [800, 500, 400, 300])
    set(gca,'TickLabelInterpreter','latex')
    hold on
    box on
    
    % colors
    orange      = [230, 135, 28]/255 ;
    KUL_blue    = [0.11,0.55,0.69] ;
    green       = [58, 148, 22]/255 ;
    red         = [194, 52, 52]/255 ;
    grey        = 0.5 * [1 1 1] ;
    purple      = [103, 78, 167]/255 ;
    
    % parameters of the plot
    prec = 1 ;
    wdw = 10 ;
    x_min = -wdw + floor( min( [ class_1(1,:), class_2(1,:) ] )) ;
    x_max =  wdw + ceil(  max( [ class_1(1,:), class_2(1,:) ] )) ;
    y_min = -wdw + floor( min( [ class_1(2,:), class_2(2,:) ] )) ;
    y_max =  wdw + ceil(  max( [ class_1(2,:), class_2(2,:) ] )) ;
    x = x_min:prec:x_max ;
    y = y_min:prec:y_max ;
    [X, Y] = meshgrid(x, y) ;           
    %

    data = [ X(:)' ; Y(:)' ] ;

%     [paths, ~, x_ends] = model.simulate( data ) ;
    
%     bin_class_1 = vecnorm(x_ends-mean(class_1, 2)) <= vecnorm(x_ends-mean(class_2, 2)) ;
%     bin_class_2 = vecnorm(x_ends-mean(class_2, 2)) <= vecnorm(x_ends-mean(class_2, 2)) ;
    
    [~, ~, x_end] = model.simulate( data ) ;
    
    dist2class1 = vecnorm(x_end-mu_1) ;
    dist2class2 = vecnorm(x_end-mu_2) ;
    dist2class3 = vecnorm(x_end-mu_3) ;
    min_dist = min(dist2class1, min(dist2class2, dist2class3)) ;
    
    bin_class1 = (dist2class1 == min_dist) ;
    bin_class2 = (dist2class2 == min_dist) ;
    bin_class3 = (dist2class3 == min_dist) ;
    
    for p = 1:size(data,2)
        c = (bin_class1(p)*orange + bin_class2(p)*green + + bin_class3(p)*purple)/(bin_class1(p)+bin_class2(p)+bin_class3(p)) ;
        scatter(data(1,p), data(2,p), 10, 'filled', 'MarkerFaceColor', c, 'MarkerFaceAlpha', .5)
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
    % class 3 patterns to memorize
    l_patterns_c3 = plot(class_3(1, :), class_3(2, :), '*', 'MarkerSize', 10, 'color', purple, 'LineWidth', 2) ;
    plot(mu_3(1, :), mu_3(2, :), '.', 'MarkerSize', 20, 'color', red, 'LineWidth', 2) ;
    

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
    %legend( [l_patterns_c1, l_patterns_c2, hlines(1)], {'Class 1', 'Class 2', 'Streamlines'}, 'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;
end


function plot_convergence(model, class_1, class_2, mu_1, mu_2)

    figure('position', [800, 100, 400, 300])
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
    prec = 2 ;
    wdw = 10 ;
    x_min = -wdw + floor( min( [ class_1(1,:), class_2(1,:) ] )) ;
    x_max =  wdw + ceil(  max( [ class_1(1,:), class_2(1,:) ] )) ;
    y_min = -wdw + floor( min( [ class_1(2,:), class_2(2,:) ] )) ;
    y_max =  wdw + ceil(  max( [ class_1(2,:), class_2(2,:) ] )) ;
    x = x_min:prec:x_max ;
    y = y_min:prec:y_max ;
    [X, Y] = meshgrid(x, y) ;           
    %

    data = [ X(:)' ; Y(:)' ] ;
    
    num_steps = 200 ;
    data_old = data ;
    data_d2class1 = zeros(size(data,2), num_steps) ;
    data_d2class2 = zeros(size(data,2), num_steps) ;
    for n = 1:num_steps 
        data_new = model.simulate_one_step( data_old ) ;
        % store step
        data_d2class1(:,n) = vecnorm(data_new-mu_1) ;
        %data_d2class2(:,n) = vecnorm(data_new-mu_2) ;
        % 
        data_old = data_new ;
    end
    
    plot(1:num_steps, data_d2class1, 'color', orange)
    %plot(1:num_steps, data_d2class2, 'color', green)

    hold off
    set(gca,'FontSize',12)
    ylim([-0.5 , max([data_d2class1, data_d2class2], [], 'all')+0.5])
    xlabel('num steps', 'interpreter', 'latex', 'fontsize', 14)
    ylabel('distance', 'interpreter', 'latex', 'fontsize', 14)
    title( "2D classification", 'interpreter', 'latex', 'fontsize', 14 )
end
