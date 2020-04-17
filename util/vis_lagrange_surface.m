% Created  by OctaveOliviers
%          on 2020-04-14 17:40:10
%
% Modified on 2020-04-16 15:26:25

function visualize_lagrange_surface( obj )

    % clc

    % only for model with 2 layers, and 2 1D-patterns to memorize
    assert( obj.num_lay == 2 )
    assert( size(obj.patterns, 1) == 1 )
    assert( size(obj.patterns, 2) == 2 )

    % extract useful parameters
    X = obj.patterns ;
    p_err = obj.p_err ;
    p_drv = obj.p_drv ;
    p_reg = ogj.p_reg ;

    W1 = obj.layers{1}.W;
    b1 = obj.layers{1}.b ;
    W2 = obj.layers{2}.W ;
    b2 = obj.layers{2}.b ;

    h1 = -3 : 0.1 : 3 ; % hidden state of first pattern
    h2 = -5 : 0.1 : 5 ; % hidden state of second pattern
    [H1, H2] = meshgrid(h1, h2) ;

    L = zeros( size(H1)) ;

    for p = 1 : length(h1)*length(h2)

        L(p) =  p_err/2 * ( (H1(p)-W1*X(1)-b1)^2 + (H2(p)-W1*X(2)-b1)^2 + (X(1)-W2*tanh(H1(p))-b2)^2 + (X(2)-W2*tanh(H2(p))-b2)^2 ) + ...
                p_drv/2 * ( W1^2 + ( ( 1+tanh(H1(p))^2 )^2 + ( 1+tanh(H2(p))^2 )^2 )*W2^2 ) + ...
                p_reg/2 * ( W1^2 + W2^2 ) ;

    end

    figure
    contour( H1, H2, L )

end