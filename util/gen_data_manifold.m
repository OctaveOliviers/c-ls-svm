% Created  by OctaveOliviers
%          on 2020-04-14 15:23:37
%
% Modified on 2020-04-14 16:26:17

% Generate data points that lie along a manifold in 2D

function data = gen_data_manifold( shape, scale, number, noise )


    switch lower(shape)

        case 'o'
            z    = exp(i*pi/1)*roots([ 1, zeros(1, number-1), 1]) ;
            data = scale*[ real(z), imag(z) ]' + noise*randn(2, number) ;

        case 'c'
            z    = exp(i*pi/1)*roots([ 1, zeros(1, 2*number-1), 1]) ;
            data = -1*scale*[ real(z(1:number)), imag(z(1:number)) ]' + noise*randn(2, number) ;

        case 's'
            c1 = gen_data_manifold( 'c', scale/2, floor(number/2), noise ) ;
            c2 = gen_data_manifold( 'c', scale/2, ceil(number/2), noise ) ;
            data = [ -c1 - [ 0 ; abs(max(c1(2, :)))] , c2 + [ 0 ; abs(min(c2(2, :)))] ] ;

        case 'x'
            c1 = gen_data_manifold( 'c', scale, floor(number/2), noise ) ;
            c2 = gen_data_manifold( 'c', scale, ceil(number/2), noise ) ;
            data = [ -c1 - [ abs(max(c1(2, :))) ; 0 ] , c2 + [ abs(min(c2(2, :))); 0 ] ] ;

        otherwise
            error( "Did not understand the shape of the manifold. Can be 'o', 's', 'u' or 'x'." )
    end
end