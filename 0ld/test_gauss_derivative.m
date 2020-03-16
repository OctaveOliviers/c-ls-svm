x = 5 ;
y = 0:0.1:10 ;
sig = 1 ;

G = exp( -(x-y).^2/2/sig^2 ) ;
E = sqrt(2)*sig*sqrt(pi)/2 * erf( (x-y)/sqrt(2)/sig ) ;

figuregauss
box on
hold on
plot(y, y-x, '-') ;
plot(y, -y+x, '-') ;
plot(y, G, '-') ;
plot(y, E, ':') ;
hold off
axis equal
grid on