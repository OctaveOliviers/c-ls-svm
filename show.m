%% Octave Oliviers - March 4th 2020

function show(varargin)
    % plot a dynamical model given its parameters
    

    figure()
    box on
    hold on
    plot(x, x, 'color', [0 0 0])
    plot(zeros(size(x)), x, 'color', [0 0 0])
    plot(x, zeros(size(x)), 'color', [0 0 0])
    plot(x, f, 'color', [0 0 1], 'linewidth', 1)
    plot(patterns, patterns, 'rx')
    hold off
    title(strcat('p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg)',', p_drv = ', num2str(p_drv) ))


    
end