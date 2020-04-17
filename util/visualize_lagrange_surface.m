% Created  by OctaveOliviers
%          on 2020-04-14 17:40:10
%
% Modified on 2020-04-16 16:31:14

function visualize_lagrange_surface( obj, H, step )

    % clc

    % only for model with 2 layers, and 2 1D-patterns to memorize
    assert( obj.num_lay == 2 )
    assert( size(obj.patterns, 1) == 1 )
    assert( size(obj.patterns, 2) == 2 )

    % extract useful parameters
    X = obj.patterns ;
    p_err = obj.layers{1}.p_err ;
    p_drv = obj.layers{1}.p_drv ;
    p_reg = obj.layers{1}.p_reg ;

    W1 = obj.layers{1}.W;
    b1 = obj.layers{1}.b ;
    W2 = obj.layers{2}.W ;
    b2 = obj.layers{2}.b ;

    H_curr = H{1} ;
    g_curr = 0.01*step{1} ;

    h1 = -0 : 0.01 : 1 ; % hidden state of first pattern
    h2 = -2 : 0.01 : -0.5 ; % hidden state of second pattern
    % h1 = H_curr(1) ;
    % h2 = H_curr(2) ;
   
    [H1, H2] = meshgrid(h1, h2) ;

    L = zeros( size(H1)) ;

    for p = 1 : length(h1)*length(h2)

        L(p) =  p_err/2 * ( ( H1(p) - W1*X(1)-b1 )^2 + ( H2(p) - W1*X(2)-b1 )^2 + ( X(1) - W2*tanh(H1(p))-b2 )^2 + ( X(2) - W2*tanh(H2(p))-b2 )^2 ) + ...
                p_drv/2 * ( (1+1)*W1^2 + ( ( 1-tanh(H1(p))^2 )^2 + ( 1-tanh(H2(p))^2 )^2 )*W2^2 ) + ...
                p_reg/2 * ( W1^2 + W2^2 ) ;

    end

    figure
    box on
    hold on
    contour( H1, H2, L )
    % surf( H1, H2, L, 'EdgeColor', 'none' )
    plot( H_curr(1), H_curr(2), 'rx')
    line( [H_curr(1) H_curr(1)-g_curr(1)], [H_curr(2) H_curr(2)-g_curr(2)] )
    hold off
    xlabel("h_1")
    ylabel("h_2")

end