% Created  by OctaveOliviers
%          on 2020-03-29 18:46:54
%
% Modified on 2020-04-11 14:51:51

classdef Memory_Model

    properties
        % model information
        name = 'Memory model'
        state   % either encoder or decoder
        % data points, map X -> Y
        X       % data points to map from
        Y       % data points to map to = target
        % model parameters
        W       % primal weights
        b       % bias
        models  % cell of shallow models in each layer
        num_lay % number of layers
        % value of lagrangian, error and jacobian
        L       % Lagrangian
        E       % constraint term 1
        J       % constraint term 2
    end

    methods
        % constructor
        function obj = Memory_Model()
            
        end


        % show update equation of model
        function show()
            % work in progress

            % f = "f = " ;
            % for l = obj.num_lay:-1:1
            %   f = join( [f, num2str(obj.W)] ) ;
            % end
        end

        
        % get weights matrix
        function w = get.W(obj)
            
            % check if can compute weights
            for l = 1:obj.num_lay
                assert( (strcmp(obj.models{l}.space, "primal" )) | ...
                        (strcmp(obj.models{l}.space, "p"      )) | ...
                        (strcmp(obj.models{l}.phi,   "poly"   )  & logical(obj.models{l}.phi==[0, 1]) ) , ...
                        "Can only compute explicit W for models trained in primal space.")
            end

            if (obj.num_lay==1)
                w = obj.W ;
            else
                w = cell(1, obj.num_lay) ;
                for l = obj.num_lay
                    w{1, l} = obj.models{l}.W ;
                end
            end
        end


        % simulate model
        function [path, varargout] = simulate(obj, start, varargin)
            % start     matrix with start positions to simulate from as columns
            % varargin  (1) array of starting values to compute update equation

            % variable to store evolution of state
            path = zeros( [size(start), 2]) ;
            path(:, :, 1) = start ;

            % initialize variables
            x_old = start ;
            x_new = simulate_one_step(obj, x_old) ;
            path(:, :, 2) = x_new ;

            % update state untill it has converged
            while (norm(x_old-x_new) >= 1e-3)
                x_old = x_new ;
                x_new = simulate_one_step(obj, x_old) ;
                path(:, :, end+1) = x_new ;

                if norm(x_new)>10*max(vecnorm(obj.X))
                    break
                end
            end

            % visualize the update map f(x) of the layer
            if (nargin>2)
                x = varargin{1} ;
                varargout{1} = obj.simulate_one_step( x ) ; ;
            end
        end


        % error of model E = Y - W' * phi(X) - B
        function E = error(obj, X, varargin)
            % X     states to compute error in

            if ( nargin<3 )
                Y = X ;
            else
                Y = varargin{1} ;

                % check correctness of input
                assert( size(X, 2)==size(Y, 2),  'Numbr of patterns in X and Y do not match.' ) ;
            end 

            E = Y - obj.simulate_one_step( X ) ;
        end


        % compute energy in state X
        function [E, varargout] = energy(obj, X)
            
            % extract usefull information
            [N, P] = size(X) ;

            E = zeros(1, P) ;
            for p = 1:P
                E(p) = -0.5 * phiTphi( X(:, p) , obj.simulate_one_step(X(:, p)), obj.phi, obj.theta ) ;
            end
        end        

        % compute energy in state X
        function [E, varargout] = energy2(obj, X)
            % X     states to compute energy, error and eigenvalues for in columns

            % extract usefull information
            [N, P] = size(X) ;

            switch obj.phi
                case { 'tanh' }
                    E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*log(cosh(X)) - 2*obj.b'*X ) ;

                
                case { 'sign' }
                    E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(obj.W)'*abs(X) - 2*obj.b'*X ) ;

                case { 'polynomial', 'poly', 'p' }
                    if ( N==1 )
                        int_k   = obj.L_e * ( ( obj.X'*X + obj.theta(2) ).^(obj.theta(1)+1) ./ obj.X' ) / (obj.theta(1)+1) ;
                        k       = obj.L_d * ( phiTphi( obj.X, X, obj.phi, obj.theta ) .* X ./ obj.X' ...
                                            - phiTphi( obj.X, X, obj.phi, [obj.theta(1)+1, obj.theta(2)] ) ./ (obj.X.^2)' / (obj.theta(1)+1) ) ;

                        E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2/obj.p_reg * (int_k + k) - 2*obj.b'*X ) ;
                    else
                        warning('do not know energy formulation yet')
                        E = zeros(1, P) ;
                    end

                case { 'gaussian', 'gauss', 'g' }
                    if ( N==1 )
                        int_k   = obj.theta*sqrt(pi/2) * obj.L_e * erfc( (obj.X' - X) / (sqrt(2)*obj.theta) ) ;
                        k       = - obj.L_d * phiTphi( obj.X, X, obj.phi, obj.theta ) ;

                        E = ( 1/2 * vecnorm(X, 2, 1).^2 - 1/obj.p_reg * (int_k + k) - obj.b'*X ) ;
                    else
                        warning('do not know energy formulation yet')
                        E = zeros(1, P) ;
                    end

                otherwise
                    warning('not exact formulation for energy yet');

                    % error term
                    err   = obj.model_error( X ) ;
                    e_kin = vecnorm( err, 2, 1 ).^2 ;

                    % derivative term
                    e_pot = zeros(1, P) ;
                    for p = 1:P
                        J = obj.jacobian( X(:, p) ) ;
                        e_pot(p) = trace( J'*J ) ;
                    end                 

                    E = obj.p_err/2 * e_kin + obj.p_drv/2 * e_pot ;
            end


            if (nargout>2)
                eig_jac = zeros( size(X) );
                for p = 1:size(X, 2)
                    eig_jac(:, p) = eig( -obj.jacobian( X(:, p) ) ) ;
                end
            end         

            if (nargout>1)
                varargout{1} = vecnorm( obj.error( X ), 2, 1 ) ;
                varargout{2} = eig_jac ;
            end
        end



        % visualize dynamical model
        function visualize(obj, varargin)
            % varargin      (1) start positions to simulate model from

            % can only visualize 1D and 2D data
            assert( size(obj.X, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

            % extract useful information
            dim_data = size(obj.X, 1) ;
            num_data = size(obj.X, 2) ;

            % if data is one dimensional, visualize update function
            if (dim_data==1)
                
                figure('position', [300, 500, 400, 300])
                
                if ( strcmp(obj.state, 'encoder') )

                    x = -1+1.5*min(obj.X, [], 'all') : ...
                        (max(obj.X, [], 'all')-min(obj.X, [], 'all'))/20/num_data : ...
                        1+1.5*max(obj.X, [], 'all') ;

                    box on
                    hold on
                    
                    % patterns to memorize
                    l_patterns = plot( obj.X, obj.X, 'rx', 'linewidth', 2 ) ;

                    % update function f(x_k)
                    f = obj.simulate_one_step(x) ;
                    l_update = plot( x, f, 'linestyle', '-', 'color', [0, 0.4470, 0.7410], 'linewidth', 1) ;

                    % simulate model from initial conditions in varargin
                    if (nargin>1)
                        x_k = varargin{1} ; 
                        p   = obj.simulate( x_k ) ;
                        
                        P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
                        P(:, :, 1:2:end) = p ;
                        P(:, :, 2:2:end) = p ;
                        P

                        for i = 1:size(P, 2)
                            plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'k-', 'linewidth', 0.5, 'linestyle', '-') ;
                        end
                        plot(squeeze(p(:, :, 1)), squeeze(p(:, :, 1)), 'kx' ) ;
                    end

                    % identity map
                    ylabel('x_{k+1}')
                    l_identity = plot(x, x, 'color', [0.4 0.4 0.4], 'linestyle', ':', 'MarkerSize', 0.01) ;

                    % yyaxis right
                    % % energy surface
                    % E = obj.energy( x ) ;
                    % l_energy = plot(x, E, 'linestyle', '-.', 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 1) ;
                    % ylabel('Energy E(x_{k})')

                    hold off
                    xlabel('x_k')
                    % axes through origin
                    % axis equal
                    ax = gca;
                    ax.XAxisLocation = 'origin';
                    % ax.YAxisLocation = 'origin';
                    title( obj.name )
                    % xlim([-4, 4])
                    % ylim([-4, 4])
                    legend( [l_patterns, l_update, l_energy, l_identity ], {'Pattern', 'Update equation', 'Energy', 'Identity map'} , 'location', 'northwest')

                else
                    % x = -1+1.5*min(obj.X, [], 'all') : ...
                    %     (max(obj.X, [], 'all')-min(obj.X, [], 'all'))/20/num_data : ...
                    %     1+1.5*max(obj.X, [], 'all') ;

                    x = -6:0.1:6 ;

                    box on
                    hold on
                    yyaxis left
                    
                    % patterns to memorize
                    l_patterns = plot( obj.X, obj.X, 'rx', 'linewidth', 2 ) ;

                    % update function f(x_k)
                    f = obj.simulate_one_step(x) ;
                    l_update = plot( x, f, 'linestyle', '-', 'color', [0, 0.4470, 0.7410], 'linewidth', 1) ;

                    % simulate model from initial conditions in varargin
                    if (nargin>1)
                        x_k = varargin{1} ; 
                        p   = obj.simulate( x_k ) ;
                        
                        P = zeros([size(p, 1), size(p, 2), 2*size(p, 3)]) ;
                        P(:, :, 1:2:end) = p ;
                        P(:, :, 2:2:end) = p ;

                        for i = 1:size(P, 2)
                            plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'k-', 'linewidth', 0.5, 'linestyle', '-') ;
                        end
                        plot(squeeze(p(:, :, 1)), squeeze(p(:, :, 1)), 'kx' ) ;
                    end

                    % identity map
                    ylabel('x_{k+1}')
                    l_identity = plot(x, x, 'color', [0.4 0.4 0.4], 'linestyle', ':', 'MarkerSize', 0.01) ;

                    % yyaxis right
                    % % energy surface
                    % E = obj.energy( x ) ;
                    % l_energy = plot(x, E, 'linestyle', '-.', 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 1) ;
                    % ylabel('Energy E(x_{k})')

                    hold off
                    xlabel('x_k')
                    % axes through origin
                    % axis equal
                    ax = gca;
                    ax.XAxisLocation = 'origin';
                    % ax.YAxisLocation = 'origin';
                    title( obj.name )
                    % xlim([-4, 4])
                    % ylim([-4, 4])
                    % legend( [l_patterns, l_update, l_energy, l_identity ], {'Pattern', 'Update equation', 'Energy', 'Identity map'} , 'location', 'northwest')                    
                end


            % if data is 2 dimensional, visualize vector field with nullclines
            elseif (dim_data==2)
            
                figure('position', [300, 500, 330, 300])

                box on
                hold on

                % patterns to memorize
                l_patterns = plot(obj.X(1, :), obj.X(2, :), 'rx', 'linewidth', 2) ;

                % energy surface and nullclines
                wdw = 10 ; % window
                prec = wdw/10 ;
                x = -wdw:prec:wdw ;
                y = -wdw:prec:wdw ;
                [X, Y] = meshgrid(x, y) ;           
                %
                F = obj.simulate_one_step( [ X(:)' ; Y(:)' ] ) ;
                f1 = reshape( F(1, :), [length(x), length(y)] ) ;
                f2 = reshape( F(2, :), [length(x), length(y)] ) ;
                % E = obj.energy( [ X(:)' ; Y(:)' ] ) ;
                % E = reshape( E, [length(x), length(y)]) ;
                quiver( X, Y, (f1-X), (f2-Y) ) ;
                %
                [~, l_nc1] = contour(x, y, X-f1,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', '--') ;
                [~, l_nc2] = contour(x, y, Y-f2,[0, 0], 'linewidth', 1, 'color', [0.5, 0.5, 0.5], 'linestyle', ':') ;
                % contour(x, y, E) ;

                % simulate model from initial conditions in varargin
                if (nargin>1)
                    x_k = varargin{1} ; 
                    p   = obj.simulate( x_k ) ;
                    
                    for i = 1:size(p, 2)
                        plot(squeeze(p(1, i, :)), squeeze(p(2, i, :)), 'color', [0 0 0], 'linewidth', 1)
                    end
                    plot(p(1, :, 1), p(2, :, 1), 'ko')
                end

                hold off
                xlabel('x_1')
                ylabel('x_2')
                xlim([-wdw, wdw])
                ylim([-wdw, wdw])
                % axes through origin
                axis equal
                % title( join([ 'p_err = ', num2str(obj.p_err), ...
                %           ', p_reg = ', num2str(obj.p_reg), ...
                %           ', p_drv = ', num2str(obj.p_drv) ]))
                % title( join([ num2str(obj.num_lay), ' layers ', obj.phi{1} ]) )
                title( obj.name )
                % legend( [l_patterns, l_nc1, l_nc2], {'Pattern', 'x_1 nullcline', 'x_2 nullcline'}, 'location', 'southwest') ;

            end
        end
    end
end