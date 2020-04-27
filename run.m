% Created  by OctaveOliviers
%          on 2020-04-11 14:54:28
%
% Modified on 2020-04-26 11:10:28

function run( varargin )

    if nargin == 0
        run_shallow
    else

        switch varargin{1}
            case 1
                run_shallow.m
            case 2
                run_deep
            otherwise
                error("Did not understand which script to run.")
        end
    end
end