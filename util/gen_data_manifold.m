% Created  by OctaveOliviers
%          on 2020-04-14 15:23:37
%
% Modified on 2020-04-27 22:07:19

% Generate data points that lie along a manifold in 2D

function data = gen_data_manifold( shape, scale, number, noise )


    switch lower(shape)

        case 'c'
            % compute the roots of unity
            z    = eigs( compan([ 1, zeros(1, 2*number-1), 1]), number, 'largestreal' ) ;
            % order them according to increasing phase
            data = -1*scale*[ real(z), imag(z) ]' + noise*randn(2, number) ;

        case '-'
            prec = 2*scale/(number-1) ;
            data = [ -scale:prec:scale ; zeros(1, number) ] + noise*randn(2, number) ;

        case {'o','0'}
            c1   = gen_data_manifold( 'c', scale, number, noise ) ;
            c2   = gen_data_manifold( 'c', scale, number, noise ) ;
            data = scale*[ real(z), imag(z) ]' + noise*randn(2, number) ;

        case 'u'
            c    = gen_data_manifold( 'c', scale, number, noise ) ;
            data = [ cos(pi/2), -sin(pi/2) ; sin(pi/2), cos(pi/2) ] * c ;

        case 's'
            c1 = gen_data_manifold( 'c', scale/2, floor(number/2), noise ) ;
            c2 = gen_data_manifold( 'c', scale/2, ceil(number/2), noise ) ;
            data = [ -c1 - [ 0 ; abs(max(c1(2, :)))] , c2 + [ 0 ; abs(min(c2(2, :)))] ] ;

        case 'x'
            c1 = gen_data_manifold( 'c', scale, floor(number/2), noise ) ;
            c2 = gen_data_manifold( 'c', scale, ceil(number/2), noise ) ;
            data = [ -c1 - [ abs(max(c1(2, :))) ; 0 ] , c2 + [ abs(min(c2(2, :))); 0 ] ] ;

        case '8'
            s1 = gen_data_manifold( 's',  scale, floor(number/2), noise ) ;
            s2 = gen_data_manifold( 's', -scale, ceil(number/2), noise ) ;
            data = [ s1 , s2 ] ;

        otherwise
            error( "Did not understand the shape of the manifold." )
    end
end