% Created  by OctaveOliviers
%          on 2020-04-16 16:40:59
%
% Modified on 2020-05-12 08:44:43

function [x_opt, varargout] = gradient_descent( objective_fcn, gradient_fcn, x_0, varargin )

    if nargin == 3
        max_itr         = 20 ;
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

        % if the gradient is small, optimization has converged
        if norm_element(grad) <= 1e-5
            break
        end

        % backtracking
        b = 0.1 ; % not unity to break symmetry
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
            break
        
        % update the solution
        else
            x_opt    = x_c ;
            obj_best = obj_new ;
        end
    end

    if nargout > 1
        varargout{1} = objective_fcn(x_opt) ;
    end
    if nargout > 2
        varargout{2} = norm_element(grad) ;
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


