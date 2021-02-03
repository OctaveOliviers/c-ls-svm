% Created  by OctaveOliviers
%          on 2020-05-15 10:50:01
%
% Modified on 2020-05-15 14:11:00

N = 2 ;
P = 2 ;
scale = 5 ;
offset = 3 ;

% create data set
X = offset + scale*randn(N, P) ;

% create sphere in data space


% project sphere in input space

% analyze data set
[U, S, V] = svd( X ) ;



% plot
figure('position', [100, 100, 300, 285])
% colors of plot
orange   = [230, 135, 28]/255 ;
KUL_blue = [0.11,0.55,0.69] ;
green    = [58, 148, 22]/255 ;
red      = [194, 52, 52]/255 ;
yellow   = [241, 194, 50]/255 ;
hold on
box on
% if 2D data
switch N

    case 2
        for p = 1:P
            line( [0, X(1, p)], [0, X(2, p)],  'color', KUL_blue, 'Linewidth', 1)
            plot( X(1, p), X(2, p), '.', 'Markersize', 15, 'color', KUL_blue )
        end
        

    case 3

hold off

end
