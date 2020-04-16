% Created  by OctaveOliviers
%          on 2020-04-14 17:40:10
%
% Modified on 2020-04-15 08:05:17

clc
clear all

N = 10 ;
D = 20 ;

W = randn(D, N) ;
F = randn(D, N) ;

% multiplication in matrix form
trace( W' * F * F' * W )

% in loop form
t = 0 ;
for d = 1:D
    for b = 1:D
        for n = 1:N
            for m = 1:N
                t = t + W(d, n) * F(d, m) * F(b, m) * W(b, n) ;
            end
        end
    end
end
t