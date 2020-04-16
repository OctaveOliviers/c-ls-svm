% Created  by OctaveOliviers
%          on 2020-03-05 09:54:32
%
% Modified on 2020-04-15 18:22:16

classdef Memory_Model

    properties
        % information about model
        patterns            % patterns to memorize
        name                % name of model
        % architecture
        num_lay             % number of layers
        layers              % cell containg each layer
        % training parameters
        max_iter            % maximum number of iterations during implicit training
        alpha               % learning rate for gradient descent in hidden states
        max_back_track = 20 % maximum number of back tracking in gradient descent
        % results of optimization process
        E                   % value of error in Lagrange function
        J                   % value of jacobian in Lagrange function
        L                   % value of Lagrange function
        H                   % hidden states of each pattern in each layer
    end

    methods

        % constructor
        function obj = Memory_Model(varargin)
            %   obj = Memory_Model( )
            %   obj = Memory_Model( layers )
            %   obj = Memory_Model( max_iter, alpha )
            %   obj = Memory_Model( layers, max_iter, alpha )

            if (nargin==0)
                obj.layers    = cell(0) ;
                obj.num_lay   = length(obj.layers) ;
                %
                obj.max_iter  = 10 ;
                obj.alpha     = .01 ;

            elseif (nargin==1)
                obj.layers    = varargin{2} ;
                obj.num_lay   = length(obj.layers) ;
                %
                obj.max_iter  = 10 ;
                obj.alpha     = .01 ;

            elseif (nargin==2)
                obj.layers    = cell(0) ;
                obj.num_lay   = length(obj.layers) ;
                %
                obj.max_iter  = varargin{1} ;
                obj.alpha     = varargin{2} ;                

            elseif (nargin==3)
                obj.layers    = varargin{1} ;
                obj.num_lay   = length(obj.layers) ;
                %
                obj.max_iter  = varargin{2} ;
                obj.alpha     = varargin{3} ; 
            end

            obj = obj.update_name() ;
        end


        % add new layer to model
        function obj = add_layer(obj, varargin)
            % either add individual layer
            %   obj.add_layer( layer )
            %
            % of add layer from its parameters
            %   obj.add_layer( n_out, space, phi, theta, p_err, p_drv, p_reg )

            % append to cell of layers
            if ( nargin < 3 )
                num_new_lay = length(varargin{1}) ;
                obj.layers(end+1:end+num_new_lay) = varargin{1} ;
            else
                switch varargin{1}
                    case {"primal", "p"}
                        obj.layers{end+1} = Layer_Primal( varargin{2:end} ) ;
                        obj.layers{end}.n_in = obj.layers{end-1}.n_out ;

                    case {"dual", "d"}
                        obj.layers{end+1} = Layer_Dual( varargin{2:end} ) ;

                    otherwise
                        error( 'Did not recognize "space" variable. Can be "primal", "p", "dual" or "d".' )
                end
            end

            % update number of layers
            obj.num_lay = length(obj.layers);
            % update name of model
            obj = obj.update_name() ;
        end


        % simulate model from start position
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

                if norm(x_new)>10*max(vecnorm(obj.patterns))
                    break
                end
            end

            % visualize the update map f(x) of the layer
            if (nargin>2)
                x = varargin{1} ;
                varargout{1} = obj.simulate_one_step( x ) ; ;
            end
        end


        % simulate model over one step
        function F = simulate_one_step(obj, X)
            % X     matrix with start positions to simulate from as columns

            F = X ;
            for l = 1:obj.num_lay
                F = obj.layers{ l }.simulate_one_step( F ) ;
            end
        end


        % compute value of error
        function E = model_error(obj, varargin)

            % compute error of model
            if (nargin<2)
                E = obj.E ;

            % compute error in new point
            else
                E = varargin{1} - obj.simulate_one_step( varargin{1} ) ;
            end
        end


        % compute value of jacobian 
        function J_new = model_jacobian(obj, varargin)
            
            % compute jacobian of model
            if (nargin==1)
                [N, P] = size( obj.patterns ) ;

            % compute jacobian in new point
            else
                [N, P] = size( varargin{1} ) ;
            end
            
            % initialize
            J_new = obj.layers{end}.layer_jacobian( varargin{:} ) ;
            % apply chain rule
            for l = obj.num_lay-1:-1:1
                % extract data of level
                J_l   = obj.layers{l}.layer_jacobian( varargin{:} ) ;
                N_l   = size(J_l, 1) ;
                N_lm1 = size(J_l, 2)/P ;
                %
                J_old = J_new ;
                J_new = zeros(N, P*N_lm1) ;
                for p = 1:P
                    J_new(:, (p-1)*N_lm1+1:p*N_lm1) = J_old(:, (p-1)*N_l+1:p*N_l) * J_l(:, (p-1)*N_lm1+1:p*N_lm1) ;
                end
            end
            J_new = (-1)^(obj.num_lay+1)*J_new ;
        end


        % compute value of Lagrange function
        function L = model_lagrangian(obj, varargin)
            
            % compute lagrangian of model
            if (nargin<2)
                L = obj.L ;

            % evaluate lagrangian with new parameters
            else
                % evaluate lagrangian in new data points
                if ~iscell(varargin{1})
                    assert( obj.num_lay==1, 'This only works for one layered models.')

                    % matrix with data points to evaluate lagrangian in
                    X = varargin{1} ;
                    [N, P] = size(X) ;

                    L = zeros(1, P) ;
                    for p = 1:P
                        L(p) = obj.layers{1}.layer_lagrangian( X(:, p), X(:, p) );
                    end

                % evaluate lagrangian with new hidden layers
                else
                    % cell of { patterns, hidden states, patterns }
                    D = varargin{1} ;
                    L = obj.layers{ 1 }.layer_lagrangian( D{1}, D{2} ) ;
                    for l = 2:obj.num_lay
                        L = L + obj.layers{ l }.layer_lagrangian( D{l}, D{l+1} ) ;
                    end
                end
            end            
        end


        % train model
        function obj = train(obj, X, varargin)
            % X         patterns to memorize

            assert( logical(obj.num_lay > 0), "Model has no layers.")

            % if one layered model
            if ( obj.num_lay==1 )
                % store patterns to memorize
                obj.patterns = X ;
                % train model
                obj.layers{1} = obj.layers{1}.train( X, varargin{:} ) ;
                % store value of Lagrange function and error
                obj.E = obj.model_error( X ) ;
                obj.J = obj.model_jacobian( X ) ;
                obj.L = obj.model_lagrangian( {X, X} ) ;

            % if deep model
            else
                if ( nargin > 2 )
                    obj = train_explicit( obj, X, varargin{1} ) ;
                else
                    obj = train_implicit( obj, X ) ;
                end
            end
        end


        % train model by explicitely assigning the hidden states
        function obj = train_explicit(obj, X, H)
            % X    patterns to memorize
            % H    cell containing hidden states
            
            % check correctness of input
            assert( length(H) == (obj.num_lay-1), ...
                    'Passed too many hidden states. Should be one less than number of layers.' ) ;

            % extract useful parameters
            [N, P]       = size(X) ;
            obj.patterns = X ;

            train_set    = [ X, H, X ] ;
            for l = 1:obj.num_lay
                obj.layers{l} = obj.layers{l}.train( train_set{l}, train_set{l+1} ) ;
            end

            % store value of Lagrange function and error
            obj.E = obj.model_error( X ) ;
            obj.J = obj.model_jacobian( X ) ;
            obj.L = obj.model_lagrangian( train_set ) ;

            disp("model trained explicitly")
        end


        % train model to implicitely find good hidden states with method of alternating coordinates
        function [obj, varargout] = train_implicit(obj, X)
            % X         patterns to memorize
            
            % extract useful parameters
            [N, P]       = size(X) ;
            obj.patterns = X ;

            % initialize hidden representations of patterns
            H            = cell( 1, obj.num_lay-1 ) ;
            % [H{:}]       = deal( X ) ;
            step         = cell( size(H) ) ;

            % initialize each hidden state randomly
            for l = 1:obj.num_lay-1
                H(l) = { randn( obj.layers{l}.N_out, P ) } ;
            end

            % keep track of evolution through parameter space
            if (nargout > 1)
                path = cell(2, obj.max_iter+2) ;
                path(1, 1:2) = { "evolution of weights",        H } ;
                path(2, 1:2) = { "evolution of hidden states",  H } ;
            end

            % train the network
            for i = 1:obj.max_iter
                i

                % train deep model using the Method of Alternating Coordinates
                % https://arxiv.org/abs/1212.5921

                % update weights with convex optimization 
                obj = obj.train_explicit( X, H ) ;
                obj.L

                % update hidden representations with gradient descent
                for l = obj.num_lay-1:-1:1

                    % objective = @(h) obj.layers{l}.lagrangian(H(:, :, l), h) + obj.layers{l}.lagrangian(h, H(:, :, l+2))
                    
                    % % objective before minimization
                    % objective(H(:, :, l+1))

                    % options = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true) ;
                    % H(:, :, l+1) = fminunc(objective, H(:, :, l+1), options) ;

                    % % objective after minimization
                    % objective(H(:, :, l+1))

                    switch obj.layers{l+1}.space

                        case {'primal', 'p'}

                            % extract useful parameters
                            E_l     = obj.layers{l}.E ;
                            E_lp1   = obj.layers{l+1}.E ;
                            J_lp1   = obj.layers{l+1}.J ;
                            W_lp1   = obj.layers{l+1}.W ;

                            % derivative at current level
                            dL_l    = obj.layers{l}.p_err * E_l ;

                            % hessian of each dimension of the feature map, evaluated in the hidden state
                            Hes     = hes( H{l}, obj.layers{l+1}.phi ) ;

                            % derivative of error wrt each hidden pattern
                            dE_lp1  = zeros(N, P) ;
                            for p = 1:P
                                dE_lp1(:, p) = J_lp1(:, 1+(p-1)*N:p*N)' * E_lp1(:, p) ;
                            end
                            % derivative of jacobian wrt each hidden pattern
                            dJ_lp1  = zeros(N, P) ;
                            for p = 1:P
                                for n = 1:N
                                    dJ_lp1(n, p) = trace( squeeze(Hes(:, n+(p-1)*N, :)) * W_lp1 * J_lp1(:, 1+(p-1)*N:p*N) ) ;
                                end
                            end
                            % derivative at next level
                            dL_lp1   = obj.layers{l+1}.p_err * dE_lp1 + obj.layers{l}.p_drv * dJ_lp1 ;

                            % store update of hidden states
                            step( l ) = { dL_l + dL_lp1 } ;
                            % step( l ) = { dL_lp1 } ;
                            
                        case {'dual', 'd'}
                            warning( 'target prop is not yet implemented for dual formulation' ) ;
                    end
                end

                % backtracking
                b = obj.alpha ;
                for k = 1:obj.max_back_track
                    % store canditate new hidden states
                    H_c = H ;
                    for l = 1:length(H)
                        H_c(l) = { cell2mat(H(l)) - b * cell2mat(step(l)) } ;
                    end

                    cell2mat(H_c)

                    % train each layer
                    
                    % check if value of lagrangian improves
                    
                    % E = obj.model_error( X ) ; % this is useless

                    % temp = obj.train_explicit( X, H_c ) ;
                    % temp.L
                    % obj.L
                    % if ( temp.L > obj.L )
                    %
                    L = obj.model_lagrangian( [ X, H_c, X ] )
                    if ( L > obj.L )
                    %
                        b = b/2 ;
                    % elseif ( temp.E == )
                    %     k = obj.max_back_track ;
                    %     break                        
                    else
                        break
                    end

                end

                % update the hidden states
                if ( k == obj.max_back_track )
                    % did not find better hidden states
                    % obj = obj.train_explicit( X, H ) ;
                    % stop training
                    disp( "found solution after " + num2str(i) + " iterations" )
                    break
                else
                    disp( "backtracking with b = " + num2str(b) + " after " + k + " backtracking steps" )
                    disp( "new value of Lagrangian = " + num2str(obj.L) )
                    disp( "new value of error = " + num2str(norm(obj.E)) )
                    disp( " " )

                    H = H_c ;

                    % keep track of evolution of parameters
                    if (nargout>1)
                        path(1:2, i+2) = { H_c, H_c } ;
                    end 
                end

                obj.visualize() ;
                pause(2)

            end

            % keep track of evolution through parameter space
            if ( nargout > 1 )
                path ;
                varargout = [path] ;
            end

            disp("model trained implicitly")
        end


        % test for energy function
        function [E, varargout] = energy_2(obj, X)

            E = obj.model_lagrangian( X ) ;

        end

        % compute energy in state X
        function [E, varargout] = energy(obj, X)
            % X     states to compute energy, error and eigenvalues for in columns

            assert( obj.num_lay == 1, "Energy function not yet implemented for deep models." )

            % extract usefull information
            [N, P] = size(X) ;

            layer = obj.layers{1} ;
            switch layer.phi
                case { 'tanh' }
                    E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(layer.W)'*log(cosh(X)) - 2*layer.b'*X ) ;

                
                case { 'sign' }
                    E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2*diag(layer.W)'*abs(X) - 2*layer.b'*X ) ;

                case { 'polynomial', 'poly', 'p' }
                    if ( N==1 )
                        int_k   = layer.L_e * ( ( layer.X'*X + layer.theta(2) ).^(layer.theta(1)+1) ./ layer.patterns' ) / (layer.theta(1)+1) ;
                        k       = layer.L_d * ( phiTphi( layer.X, X, layer.phi, layer.theta ) .* X ./ layer.X' ...
                                            - phiTphi( layer.X, X, layer.phi, [layer.theta(1)+1, layer.theta(2)] ) ./ (layer.X.^2)' / (layer.theta(1)+1) ) ;

                        E = 1/2 * ( vecnorm(X, 2, 1).^2 - 2/layer.p_reg * (int_k + k) - 2*layer.b'*X ) ;
                    else
                        warning('do not know energy formulation yet')
                        E = zeros(1, P) ;
                    end

                case { 'gaussian', 'gauss', 'g' }
                    if ( N==1 )
                        int_k   = layer.theta*sqrt(pi/2) * layer.L_e * erfc( (layer.X' - X) / (sqrt(2)*layer.theta) ) ;
                        k       = - layer.L_d * phiTphi( layer.X, X, layer.phi, layer.theta ) ;

                        E = ( 1/2 * vecnorm(X, 2, 1).^2 - 1/layer.p_reg * (int_k + k) - layer.b'*X ) ;
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
                        J = obj.model_jacobian( X(:, p) ) ;
                        e_pot(p) = trace( J'*J ) ;
                    end                 

                    E = obj.p_err/2 * e_kin + obj.p_drv/2 * e_pot ;
            end

            if (nargout>2)
                eig_jac = zeros( size(X) );
                for p = 1:size(X, 2)
                    eig_jac(:, p) = eig( -obj.model_jacobian( X(:, p) ) ) ;
                end
            end         

            if (nargout>1)
                varargout{1} = vecnorm( obj.model_error( X ), 2, 1 ) ;
                varargout{2} = eig_jac ;
            end
        end


        % visualize dynamical model
        function visualize(obj, varargin)
            % varargin      (1) start positions to simulate model from

            % can only visualize 1D and 2D data
            assert( size(obj.patterns, 1)<3 , 'Cannot visualize more than 2 dimensions.' ) ;

            % extract useful information
            dim_data = size(obj.patterns, 1) ;
            num_data = size(obj.patterns, 2) ;

            % create figure box
            figure('position', [300, 500, 330, 300])
            box on
            hold on

            % if data is one dimensional, visualize update function
            if (dim_data==1)
                
                % x = -1+1.5*min(obj.X, [], 'all') : ...
                %     (max(obj.X, [], 'all')-min(obj.X, [], 'all'))/20/num_data : ...
                %     1+1.5*max(obj.X, [], 'all') ;

                wdw = 10 ;
                prec = wdw/25 ;
                x = -wdw:prec:wdw ;

                yyaxis left
                
                % update function f(x_k)
                f = obj.simulate_one_step(x) ;
                l_update = plot( x, f, 'linestyle', '-', 'color', [0, 0.4470, 0.7410], 'linewidth', 1) ;

                % draw derivative to update equation
                % J = -1*obj.model_jacobian( x ) ;
                % for p = 1:length(x)
                %     line( [ x(p), x(p)+prec ], [ f(p), f(p)+prec*J(p) ], 'color', [0,0,1] )
                % end

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

                % patterns to memorize
                l_patterns = plot( obj.patterns, obj.patterns, 'rx', 'linewidth', 2 ) ;

                % yyaxis right
                % % energy surface
                % E = obj.energy_2( x ) ;
                % l_energy = semilogy(x, E, 'linestyle', '-.', 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 1) ;
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

            % if data is 2 dimensional, visualize vector field with nullclines
            elseif (dim_data==2)
            
                % energy surface and nullclines
                wdw = 20 ; % window
                prec = wdw/20 ;
                x = -wdw:prec:wdw ;
                y = -wdw:prec:wdw ;
                [X, Y] = meshgrid(x, y) ;           
                %
                F = obj.simulate_one_step( [ X(:)' ; Y(:)' ] ) ;
                f1 = reshape( F(1, :), [length(x), length(y)] ) ;
                f2 = reshape( F(2, :), [length(x), length(y)] ) ;
                % E = obj.energy( [ X(:)' ; Y(:)' ] ) ;
                % E = reshape( E, [length(x), length(y)]) ;
                scale = 0.5 ;
                % quiver( X, Y, (f1-X), (f2-Y), scale ) ;
                % contour(x, y, E) ;

                % plot stream lines
                [~, on] = inpolygon( X(:), Y(:), [min(x), max(x), max(x), min(x)], [min(y), min(y), max(y), max(y)] ) ;
                % streamline( X, Y, (f1-X), (f2-Y), X(on), Y(on) )
                streamline( X, Y, (f1-X), (f2-Y), X(1:4:end), Y(1:4:end) )

                % simulate model from initial conditions in varargin
                if (nargin>1)
                    x_k = varargin{1} ; 
                    p   = obj.simulate( x_k ) ;
                    
                    for i = 1:size(p, 2)
                        plot(squeeze(p(1, i, :)), squeeze(p(2, i, :)), 'color', [0 0 0], 'linewidth', 1)
                    end
                    plot(p(1, :, 1), p(2, :, 1), 'ko')
                end

                % draw principal component of jacobian in each grid point
                J = obj.model_jacobian( [ X(:)' ; Y(:)' ] ) ;
                E_x = zeros(size(X)) ;
                E_y = zeros(size(Y)) ;
                div = zeros(size(X)) ;
                for p = 1:size(J, 2)/2
                    [eig_vec, eig_val] = eigs( J(:, 1+2*(p-1):2*p), 1 ) ;
                    E_x(p) = abs(eig_val)*eig_vec(1) ;
                    E_y(p) = abs(eig_val)*eig_vec(2) ;
                    % div(p) = trace( J(:, 1+2*(p-1):2*p) ) ;
                end
                % % arrows in one direction
                % quiver(X, Y,  E_x,  E_y, scale )
                % % arrows in the other direction
                % quiver(X, Y, -E_x, -E_y, scale )
                % div = divergence( X, Y, (f1-X), (f2-Y) ) ;
                % contour( X, Y, div )

                %
                [~, l_nc1] = contour(x, y, X-f1,[0, 0], 'linewidth', 1, 'color', [0.2, 0.2, 0.2], 'linestyle', '--') ;
                [~, l_nc2] = contour(x, y, Y-f2,[0, 0], 'linewidth', 1, 'color', [0.2, 0.2, 0.2], 'linestyle', ':') ;

                % patterns to memorize
                l_patterns = plot(obj.patterns(1, :), obj.patterns(2, :), 'rx', 'linewidth', 2) ;

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


        % update name of model
        function obj = update_name(obj)
            obj.name = join([ num2str(obj.num_lay), '-layered network (']) ;
            for l = 1:obj.num_lay
                obj.name = append( obj.name, join([obj.layers{l}.phi, ', ']) ) ;
            end
            obj.name = append( obj.name(1:end-2), ')' ) ;
        end
    end
end