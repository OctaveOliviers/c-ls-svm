function plot_1d_map(patterns, x, f, varargin)

    % create figure box
    figure('position', [500, 700, 300, 285])
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
    box on
    hold on

    % colors
    orange = [230, 135, 28]/255 ;
    KUL_blue = [0.11,0.55,0.69] ;
    green = [58, 148, 22]/255 ;
    red = [194, 52, 52]/255 ;

    % plot axis
    line( [min(x) max(x)], [0 0], 'color', [0.8, 0.8, 0.8], 'linewidth', 0.5)
    line( [0 0], [min(x) max(x)], 'color', [0.8, 0.8, 0.8], 'linewidth', 0.5)

    % update function f(x_k)
    l_update = plot( x, f, 'linestyle', '-', 'color', KUL_blue, 'linewidth', 1) ;

%     % simulate model from initial conditions in varargin
%     if (nargin>1) && ~isempty(varargin{1})
%     p = obj.simulate( varargin{1} ) ;

%     for i = 1:length(p)
% 
%         P = zeros([1, 2*size(p{i},2)]) ;
%         P(:, 1:2:end) = p{i} ;
%         P(:, 2:2:end) = p{i} ;
% 
%         plot(P(:,1:end-1), P(:,2:end), 'linewidth', 1, 'linestyle', '-','color', orange) ;
%         plot(p{i}(1, 1), p{i}(1, 1), 'o', 'color', orange, 'linewidth', 1.5 ) ;
%     end

    % identity map
    l_identity = plot(x, x, 'color', [0.4 0.4 0.4], 'linestyle', ':') ;

    % patterns to memorize
    l_patterns = plot( patterns, patterns, '.', 'MarkerSize', 15, 'color', orange ) ;

    hold off
    set(gca,'FontSize',12)
%     xlabel('$x^{(k)}$', 'interpreter', 'latex', 'fontsize', 14)
%     ylabel('$x^{(k+1)}$', 'interpreter', 'latex', 'fontsize', 14)
    xlabel('$x^{(0)}$', 'interpreter', 'latex', 'fontsize', 14)
    ylabel('$x^{(\infty)}$', 'interpreter', 'latex', 'fontsize', 14)
    
    xlim([ min(x) max(x) ])
    ylim([ min(x) max(x) ])
    % title( obj.name,'interpreter', 'latex', 'fontsize', 14 )
%     legend( [l_patterns ], ...
%               {'pattern'} , ...
%               'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12)

end