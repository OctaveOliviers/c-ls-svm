%% Octave Oliviers - March 4th 2020

function show(patterns, formulation, varargin)
    % plot a dynamical model given its parameters
    
    % extract useful information
    dim_data = size(patterns, 1) ;
    num_data = size(patterns, 2) ;


    figure()

    if (dim_data==1)

    	box on
	    hold on
	    plot(x, x, 'color', [0 0 0])
	    plot(zeros(size(x)), x, 'color', [0 0 0])
	    plot(x, zeros(size(x)), 'color', [0 0 0])
	    plot(x, f, 'color', [0 0 1], 'linewidth', 1)
	    plot(patterns, patterns, 'rx')
	    hold off
	    title(strcat('p_err = ', num2str(p_err),', p_reg = ', num2str(p_reg)',', p_drv = ', num2str(p_drv) ))
    
	elseif (dim_data==2)
    

    else
    	error('Cannot visualize more than 2 dimensions.')
    end




    
end