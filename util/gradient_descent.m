% Created  by OctaveOliviers
%          on 2020-04-16 16:40:59
%
% Modified on 2020-04-26 11:05:05

function x_opt = gradient_descent( objective_fcn, gradient_fcn, x_0, varargin )

    if nargin == 3
        max_itr         = 10 ;
        max_back_track  = 15 ;
    elseif nargin == 5
        max_itr         = varargin{1} ;
        max_back_track  = varargin{2} ;
    end

    % hold current objective value ;
    obj_best = objective_fcn( x_0 ) ;
    % initialize optimal solution
    x_opt = x_0 ;
    % start gradient descent steps
    for i = 1:max_itr
       
        grad = gradient_fcn( x_opt ) ;

        % if the gradient is too small, optimization has converged
        if norm_element(grad) <= 1e-5
            disp( "Gradient descent" )
            disp( "found solution after " + num2str(i) + " iterations" )
            disp( "with gradident size  =  " + num2str( norm_element(grad) ) )
            disp( "with objective value = " + num2str( objective_fcn(x_opt) ) )
            disp( "  " )
            break
        end

        % backtracking
        b = 0.001 ; % not unity to break symmetry
        for k = 1:max_back_track
            % store canditate new solution
            x_c = update_solution( x_opt, b, grad ) ;

            % check if objective value improves
            obj_new = objective_fcn( x_c ) ;
            if ( obj_new > obj_best )
                b = b/2 ;                  
            else
                break
            end

        end

        % did not find better solution
        if ( k == max_back_track )
            disp( "Gradient descent" )
            disp( "found solution after " + num2str(i) + " iterations" )
            disp( "with gradident size  =  " + num2str( norm_element(grad) ) )
            disp( "with objective value = " + num2str( objective_fcn(x_opt) ) )
            disp( "  " )
            break
        
        % update the solution
        else
            x_opt    = x_c ;
            obj_best = obj_new ;

            disp( "backtracking with b = " + num2str(b) + " after " + k + " backtracking steps" )
            disp( "new objective value = " + num2str( objective_fcn(x_opt) ) )
            disp( " " )
        end
    end
end


function n = norm_element(e)

    % element is a matrix or vector
    if ~iscell(e)
        n = norm(e, 'fro') ;
    
    % element is a cell 
    else
        n = 0 ;
        for l = 1:length(e)
            n = n + norm( cell2mat(e(l)), 'fro') ;
        end
    end
end


function x_new = update_solution( x, b, grad )

    % x is a matrix or vector
    if ~iscell(x)
        x_new = x - b * grad ;
    
    % x is a cell 
    else
        % allocate memory
        x_new = x ;
        % update x
        for l = 1:length(x)
            x_new(l) = { cell2mat(x(l)) - b * cell2mat(grad(l)) } ;
        end
    end
end