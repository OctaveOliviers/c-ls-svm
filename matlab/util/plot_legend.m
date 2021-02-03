% Created  by OctaveOliviers
%          on 2020-05-05 09:41:45
%
% Modified on 2020-06-02 19:37:52

% colors
orange = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green = [58, 148, 22]/255 ;
red = [194, 52, 52]/255 ;

% create figure box
% figure('position', [300, 500, 300, 285])
figure('position', [300, 500, 300, 285])
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
hold on

% l_identity = line([nan nan], [nan nan], 'color', [0.4 0.4 0.4], 'linestyle', ':') ;
% l_update = line([nan nan], [nan nan], 'linestyle', '-', 'color', KUL_blue, 'linewidth', 1) ;
% l_patterns = plot( nan, nan, '.', 'MarkerSize', 20, 'color', orange ) ;

l_stored = line([nan nan], [nan nan], 'color', KUL_blue, 'linestyle', '-', 'linewidth', 1) ;
l_recon  = line([nan nan], [nan nan], 'linestyle', ':', 'color', orange, 'linewidth', 1.5) ;
p_start  = plot( nan, nan, '.', 'MarkerSize', 20, 'color', orange ) ;
p_end    = plot( nan, nan, 'x', 'MarkerSize', 8, 'LineWidth', 2, 'color', orange ) ;

hold off

% legend([l_patterns, l_update, l_identity ], ...
%         {'Memory', 'Update equation', 'Identity map'}, ...
%         'location', 'best', 'interpreter', 'latex', 'fontsize', 12);
% title( "Legend",'interpreter', 'latex', 'fontsize', 14 )

legend( [l_stored, l_recon, p_start, p_end], ...
        {'Movement to memorize', 'Reconstructed movement', 'Start', 'End'}, ...
        'location', 'best','interpreter', 'latex', 'fontsize', 12) ;

% set(l_patterns, 'visible', 'off');
% set(l_identity, 'visible', 'off');
% set(l_update, 'visible', 'off');
set(gca, 'visible', 'off');