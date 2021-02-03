% Created  by OctaveOliviers
%          on 2020-05-06 09:59:42
%
% Modified on 2020-06-01 21:57:13

% Make histogram of singular values of jacobian
function plot_singular_values( sv )
    % sv    matrix of size (k x P)      k singular values for P points

    % colors
    orange = [230, 135, 28]/255 ;
    KUL_blue = [0.11,0.55,0.69] ;
    green = [58, 148, 22]/255 ;
    red = [194, 52, 52]/255 ;
    C = [ green ; red ; KUL_blue ; orange ] ;

    % create figure box
    figure('position', [300, 500, 300, 285])
    set(gcf,'renderer','Painters')
    % figure('position', [300, 500, 170, 160])
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
    box on
    hold on
    
    for k = 1:size(sv, 1)
        % histogram of the alrgest singular values
        h = histogram( sv(k, :) , 'BinWidth', 0.1, 'FaceColor', C(k, :), 'EdgeAlpha', 0 ) ;
        x = h.BinEdges ;
        y = h.Values ;
        text( x(1:end-1)+0.05, y, num2str(y'), 'vert','bottom','horiz','center', 'interpreter', 'latex', 'FontSize',12); 
    end
    
    hold off
    set(gca,'FontSize',12)
    xlabel('Value of $\sigma_1$ (green) and $\sigma_2$ (red)', 'interpreter', 'latex', 'fontsize', 14)
    ylabel('Count', 'interpreter', 'latex', 'fontsize', 14)
    yticks([])
    % xlim([0, 1])

end