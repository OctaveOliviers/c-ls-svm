
iter = 10 ;
E = zeros(iter, 1) ;

slope = -1.5 ;

x = 1 ;



for i = 2:iter
    x_prev = x ;
    x      = slope*x_prev ;
    x_next = slope*x ;
    
    E(i) = (x_next-x)^2 + (x_prev-x)*(x_next-x) ;
    
end

E


figure()
box on
hold on
plot(x, x, 'color', [0 0 0])
plot(zeros(size(x)), x, 'color', [0 0 0])
plot(x, zeros(size(x)), 'color', [0 0 0])
plot(x, W'*tanh(x)+b, 'color', [0 0 1], 'linewidth', 1)
plot(patterns, patterns, 'rx')
hold off
title(strcat('eta = ', num2str(eta),', gamma = ', num2str(gamma) ))