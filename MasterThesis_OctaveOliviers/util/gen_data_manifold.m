% Created  by OctaveOliviers
%          on 2020-04-14 15:23:37
%
% Modified on 2020-05-08 21:07:46

% Generate data points that lie along a manifold in 2D

function data = gen_data_manifold( shape, scale, number, noise )


    switch lower(shape)

        case 'c'
            % compute the roots of unity
            z    = eigs( compan([ 1, zeros(1, 2*number-1), 1]), number, 'largestreal' ) ;
            % sort according to imaginary part
            z    = sort(z, 'ComparisonMethod', 'real') ;
            [~, idx] = sort(imag(z));
            z    = z(idx);
            % scale and add noise
            data = -1*scale*[ real(z), imag(z) ]' + noise*randn(2, number) ;

        case '-'
            prec = 2*scale/(number-1) ;
            data = [ -scale:prec:scale ; zeros(1, number) ] + noise*randn(2, number) ;        

        case {'o','0'}
            c   = gen_data_manifold( 'c', scale, number, noise ) ;
            data = [ -c , c ] ;

        case 'u'
            c    = gen_data_manifold( 'c', scale, number, noise ) ;
            data = [ cos(pi/2), -sin(pi/2) ; sin(pi/2), cos(pi/2) ] * c ;

        case 's'
            c = gen_data_manifold( 'c', scale/2, floor(number/2), noise ) ;
            data = [ c + [ 0 ; abs(min(c(2, :)))] , fliplr(-c - [ 0 ; abs(min(c(2, :)))]) ] ;

        case 'x'
            c1 = gen_data_manifold( 'c', scale, floor(number/2), noise ) ;
            c2 = gen_data_manifold( 'c', scale, ceil(number/2), noise ) ;
            data = [ -c1 - [ abs(max(c1(2, :))) ; 0 ] , c2 + [ abs(min(c2(2, :))); 0 ] ] ;

        case '8'
            s = gen_data_manifold( 's', -scale, ceil(number/2), noise ) ;
            data = [ s , -s ] ;

        case 'star'
            l = gen_data_manifold( '-', scale/2, ceil(number/5), noise ) ;
            l = l + [ min(l(1, :)) ; 0] ;

            data = [ [ cos(0*pi/2.5), -sin(0*pi/2.5) ; sin(0*pi/2.5), cos(0*pi/2.5) ]*l ...
                     [ cos(1*pi/2.5), -sin(1*pi/2.5) ; sin(1*pi/2.5), cos(1*pi/2.5) ]*l ...
                     [ cos(2*pi/2.5), -sin(2*pi/2.5) ; sin(2*pi/2.5), cos(2*pi/2.5) ]*l ...
                     [ cos(3*pi/2.5), -sin(3*pi/2.5) ; sin(3*pi/2.5), cos(3*pi/2.5) ]*l ...
                     [ cos(4*pi/2.5), -sin(4*pi/2.5) ; sin(4*pi/2.5), cos(4*pi/2.5) ]*l ] ;

        case {'concentric', 'cc'}
            o1 = gen_data_manifold( 'o', scale, ceil(2*number/3), noise ) ;
            o2 = gen_data_manifold( 'o', scale/2, ceil(number/3), noise ) ;
            data = [ o1, o2 ] ;

        case {'grouped', 'g'}
            g = [scale; 0] + noise*randn(2, ceil(number/8)) ;
            data = [ [ cos(0*pi/4), -sin(0*pi/4) ; sin(0*pi/4), cos(0*pi/4) ]*g ...
                     [ cos(1*pi/4), -sin(1*pi/4) ; sin(1*pi/4), cos(1*pi/4) ]*g ...
                     [ cos(2*pi/4), -sin(2*pi/4) ; sin(2*pi/4), cos(2*pi/4) ]*g ...
                     [ cos(3*pi/4), -sin(3*pi/4) ; sin(3*pi/4), cos(3*pi/4) ]*g ...
                     [ cos(4*pi/4), -sin(4*pi/4) ; sin(4*pi/4), cos(4*pi/4) ]*g ...
                     [ cos(5*pi/4), -sin(5*pi/4) ; sin(5*pi/4), cos(5*pi/4) ]*g ...
                     [ cos(6*pi/4), -sin(6*pi/4) ; sin(6*pi/4), cos(6*pi/4) ]*g ...
                     [ cos(7*pi/4), -sin(7*pi/4) ; sin(7*pi/4), cos(7*pi/4) ]*g ] ;

        otherwise
            error( "Did not understand the shape of the manifold." )
    
    end
end