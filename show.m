% @Author: OctaveOliviers
% @Date:   2020-03-05 19:26:18
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-06 08:46:30

function show(patterns, formulation, varargin)
    % plot a dynamical model given its parameters
    
    % can only visualize 1D and 2D data
    assert( size(patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

    % extract useful information
    dim_data = size(patterns, 1) ;
    num_data = size(patterns, 2) ;

    % check if network memorized patterns or movements
    if ( ndims(patterns)==3 )
        len_data = size(patterns, 3) ;
        bool_move = true ;
    end




    figure()

    % if data is one dimensional, visualize update function
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
    
    % else if data is 2 dimensional, visualize vector field with nullclines
	elseif (dim_data==2)
    

    else
    	error()
    end




    
end