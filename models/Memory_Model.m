% Created  by OctaveOliviers
%          on 2020-03-05 09:54:32
%
% Modified on 2020-05-08 09:34:50

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
        max_back_track = 30 % maximum number of back tracking in gradient descent
        max_opt        = 20 % maximum number of newton steps to update hidden states
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
                obj.layers    = varargin{1} ;
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
            %   obj.add_layer( n_out, p_err, p_drv, p_reg, phi, theta )

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
        function [F, varargout] = simulate_one_step(obj, X)
            % X     matrix with start positions to simulate from as columns

            if nargout == 1
                F = X ;
                for l = 1:obj.num_lay
                    F = obj.layers{ l }.simulate_one_step( F ) ;
                end
            else
                % cell to store hidden state in each layer
                states = cell( 1, obj.num_lay+1 ) ;
                states{1} = { X } ;
                for l = 1:obj.num_lay
                    states{l+1} = { obj.layers{ l }.simulate_one_step( cell2mat(states{l}) ) } ;
                end
                F = states{end} ;
                varargout{1} = states ;
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
            
            % pass input through network to get state at each level
            [~, states] = obj.simulate_one_step( varargin{1} ) ;

            % initialize
            J_new = obj.layers{end}.layer_jacobian( cell2mat(states{end-1}) ) ;
            % apply chain rule
            for l = obj.num_lay-1:-1:1
                % extract data of level
                J_l   = obj.layers{l}.layer_jacobian( cell2mat(states{l}) ) ;
                N_l   = size(J_l, 1) ;
                N_lm1 = size(J_l, 2)/P ;
                %
                J_old = J_new ;
                J_new = zeros(N, P*N_lm1) ;
                for p = 1:P
                    J_new(:, (p-1)*N_lm1+1:p*N_lm1) = J_old(:, (p-1)*N_l+1:p*N_l) * J_l(:, (p-1)*N_lm1+1:p*N_lm1) ;
                end
            end
            J_new = J_new ;
        end


        % compute value of Lagrange function
        function L = model_lagrangian(obj, varargin)
            
            % compute lagrangian of model
            if nargin == 1
                L = obj.L ;

            % evaluate lagrangian with new parameters
            elseif nargin == 2
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
                    % varargin is cell of { hidden states }
                    D = [ obj.patterns, varargin{1}, obj.patterns ] ;
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

            assert( obj.num_lay >= 1, "Model has no layers.")

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
                    % varargin{1} contains the explicit hidden states
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
            % obj.L = obj.model_lagrangian( train_set ) ;
            obj.L = obj.model_lagrangian( H ) ;

            % disp("model trained explicitly")
        end


        % train model to implicitely find good hidden states with method of alternating coordinates
        function [obj, varargout] = train_implicit(obj, X)
            % X         patterns to memorize
            
            % extract useful parameters
            [N, P]       = size(X) ;
            obj.patterns = X ;

            % initialize hidden representations of patterns
            H = cell( 1, obj.num_lay-1 ) ;
            
            % initialize each hidden state randomly
            % [H{:}] = deal( X ) ;
            for l = 1:obj.num_lay-1
               H(l) = { randn( obj.layers{l}.N_out, P ) } ;
            end

            % keep track of evolution through parameter space
            if (nargout > 1)
                path = cell(2, obj.max_iter+2) ;
                path(1, 1:2) = { "evolution of weights",        H } ;
                path(2, 1:2) = { "evolution of hidden states",  H } ;
            end

            % keep track of evolution
            fprintf( '          %10s   %7s      %7s  \n','Lagrange', 'sec.', 'grad. norm' ) ;
            fprintf( '            --------      -----     ---------- \n') ;

            % train the network
            L_old = inf ;
            for i = 1:obj.max_iter

                % train deep model using the Method of Alternating Coordinates
                % https://arxiv.org/abs/1212.5921

                % update weights with convex optimization
                fprintf( 'W-step') ; tic
                %
                obj = obj.train_explicit( X, H ) ;
                %
                fprintf( '      %6.2e      %2.2f \n', obj.model_lagrangian( H ), toc ) ; 

                % update hidden representations with gradient descent
                fprintf( 'H-step') ; tic
                %
                [H, L, g] = gradient_descent( @obj.model_lagrangian, @obj.gradient_lagrangian_wrt_H, H, obj.max_opt, obj.max_back_track ) ;
                %
                fprintf( '      %6.2e      %2.2f       %6.2e \n\n', L, toc, g ) ;

                % check for convergence
                if abs(L-L_old) < 1e-3
                    break
                end
                L_old = L ;

                % obj.visualize() ;
                % pause(2)

            end

            % keep track of evolution through parameter space
            if ( nargout > 1 )
                path ;
                varargout = [path] ;
            end

            % disp("model trained implicitly")
        end

        
        % compute gradient of Lagrange function wrt hidden states
        function grad = gradient_lagrangian_wrt_H(obj, H)
            % allocate memory
            grad = H ;
            
            % compute gradient wrt each hidden state
            for l = obj.num_lay-1:-1:1
                % gradient wrt layer l
                dL_l   = obj.layers{l}.gradient_lagrangian_wrt_output() ;
                % gradient wrt layer l+1
                dL_lp1 = obj.layers{l+1}.gradient_lagrangian_wrt_input() ;
                
                grad( l ) = { dL_l + dL_lp1 } ;                
            end
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
                                            -   phiTphi( layer.X, X, layer.phi, [layer.theta(1)+1, layer.theta(2)] ) ./ (layer.X.^2)' / (layer.theta(1)+1) ) ;

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


        % generate data points
        function samples = generate(obj, x_0, num, sig)

            % extract parameters
            N = size(x_0, 1) ;

            samples = zeros(N, num) ;
            % generate
            for t = 1:num
                % create random variation
                noise = sig*randn(N, 1) ;
                % compute model jacobian
                J = obj.model_jacobian( x_0 ) ;
                % move along manifold
                samples(:, t) = obj.simulate_one_step( x_0 ) + J*noise;

                x_0 = samples(:, t) ;
            end

        end


        % compute k largest singular triplets of jacobian in a set of points
        function [U, S] = jacobian_singular_values(obj, varargin)

            % all singular triplets of stored patterns
            if nargin == 1
                J = obj.J ;
                [N, P] = size( obj.patterns ) ;
                k = N ;
            % only k largest singular triplets of stored patterns
            elseif nargin == 2
                J = obj.J ;
                [N, P] = size( obj.patterns ) ;
                k = varargin{1} ;
            % only k largest singular triplets innew data points
            elseif nargin == 2
                J = obj.J ;
                [N, P] = size( varargin{2} ) ;
                k = varargin{1} ;
            end

            % singular values
            S = zeros( k, P ) ;
            % left singular vectors
            U = zeros( N, P, k ) ;
            
            for p = 1:P
                % compute singular vectors and singular values
                [Uk, Sk, ~] = svds( J(:, 1+N*(p-1):N*p), k, 'largest' ) ;
                % store k singular values
                S( :, p ) = diag(Sk) ;
                % store k left singular vectors
                U( :, p, : ) = Uk ;
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

            % colors
            orange = [230, 135, 28]/255 ;
            KUL_blue = [0.11,0.55,0.69] ;
            green = [58, 148, 22]/255 ;
            red = [194, 52, 52]/255 ;

            % create figure box
            figure('position', [300, 500, 300, 285])
            % figure('position', [300, 500, 170, 160])
            set(groot, 'DefaultAxesTickLabelInterpreter', 'latex')
            box on
            hold on

            % if data is one dimensional, visualize update function
            if (dim_data==1)
                
                % x = -1+1.5*min(obj.X, [], 'all') : ...
                %     (max(obj.X, [], 'all')-min(obj.X, [], 'all'))/20/num_data : ...
                %     1+1.5*max(obj.X, [], 'all') ;

                wdw = 10 ;
                prec = wdw/40 ;
                x = -wdw:prec:wdw ;

                % yyaxis left
                
                % plot axis
                line( [-wdw wdw], [0 0], 'color', [0.8, 0.8, 0.8], 'linewidth', 0.5)
                line( [0 0], [-wdw wdw], 'color', [0.8, 0.8, 0.8], 'linewidth', 0.5)

                % update function f(x_k)
                f = obj.simulate_one_step(x) ;
                l_update = plot( x, f, 'linestyle', '-', 'color', KUL_blue, 'linewidth', 1) ;

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
                        plot(squeeze(P(1, i, 1:end-1)), squeeze(P(1, i, 2:end)), 'linewidth', 1, 'linestyle', '-','color', orange) ;
                    end
                    plot(squeeze(p(:, :, 1)), squeeze(p(:, :, 1)), 'o', 'color', orange, 'linewidth', 1.5 ) ;
                end

                % identity map
                ylabel('$x^{(k+1)}$', 'interpreter', 'latex', 'fontsize', 14)
                l_identity = plot(x, x, 'color', [0.4 0.4 0.4], 'linestyle', ':') ;

                % patterns to memorize
                l_patterns = plot( obj.patterns, obj.patterns, '.', 'MarkerSize', 15, 'color', orange ) ;

                % yyaxis right
                % % energy surface
                % E = obj.energy_2( x ) ;
                % l_energy = semilogy(x, E, 'linestyle', '-.', 'color', [0.8500, 0.3250, 0.0980], 'linewidth', 1) ;
                % ylabel('Energy $E(x)$', 'interpreter', 'latex', 'fontsize', 14)

                hold off
                set(gca,'FontSize',12)
                xlabel('$x^{(k)}$', 'interpreter', 'latex', 'fontsize', 14)

                % axes through origin
                % axis equal
                % ax = gca;
                % ax.XAxisLocation = 'origin';
                % ax.YAxisLocation = 'origin';
                xlim([ -wdw wdw ])
                ylim([ -wdw wdw ])
                % title( obj.name,'interpreter', 'latex', 'fontsize', 14 )
                % xlim([-4, 4])
                % ylim([-4, 4])
                % legend( [l_patterns, l_update, l_energy, l_identity ], ...
                %           {'Pattern', 'Update equation', 'Energy', 'Identity map'} , ...
                %           'location', 'northwest', 'interpreter', 'latex', 'fontsize', 12)

            % if data is 2 dimensional, visualize vector field with nullclines
            elseif (dim_data==2)
            
                % energy surface and nullclines
                wdw = 20 ; % window
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

                % scale = 1 ;
                % % normalize the vectors
                % U = (f1-X) ./ vecnorm( f1-X ) ;
                % V = (f2-Y) ./ vecnorm( f2-Y ) ;
                % qui = quiver( X, Y, U, V ) ;
                % set(qui,'LineWidth',1,'Color', KUL_blue)
                % hs = get(qui,'MaxArrowSize');
                % set(qui,'MaxArrowSize',hs/10)
                % contour(x, y, E) ;

                % plot stream lines
                [~, on] = inpolygon( X(:), Y(:), [min(x), max(x), max(x), min(x)], [min(y), min(y), max(y), max(y)] ) ;
                % hlines = streamline( X, Y, (f1-X), (f2-Y), X(on), Y(on) ) ;
                hlines = streamslice( X, Y, (f1-X), (f2-Y), 0.5) ;
                % hlines = streamline( X, Y, (f1-X), (f2-Y), X(1:2:end), Y(1:2:end)) ;
                set(hlines,'LineWidth',0.5,'Color', KUL_blue)

                % simulate model from initial conditions in varargin
                if (nargin==2) && ~isempty(varargin{1})
                    x_k = varargin{1} ; 
                    p   = obj.simulate( x_k ) ;
                    
                    for i = 1:size(p, 2)
                        plot(squeeze(p(1, i, :)), squeeze(p(2, i, :)), 'color', orange, 'linewidth', 1)
                    end
                    plot(p(1, :, 1), p(2, :, 1), 'o', 'color', orange, 'linewidth', 1.5)
                end

                % plot manifold of the data
                if (nargin>=3) && ~isempty(varargin{2})
                    manifold = varargin{2} ;
                    l_man = plot(manifold(1, :), manifold(2, :), '--', 'color', [0.5, 0.5, 0.5], 'linewidth', 1) ;
                end

                % draw principal component of jacobian in each grid point
                [U, S] = obj.jacobian_singular_values( ) ; 
                scale = 3 ;
                % arrows in direction largest component
                l_J_1 = quiver( obj.patterns(1, :), obj.patterns(2, :), scale*U(1, :, 1), scale*U(2, :, 1), 'AutoScale', 'off' ) ;
                set(l_J_1,'LineWidth',2,'Color', green, 'ShowArrowHead', 'off')
                l_J_2 = quiver( obj.patterns(1, :), obj.patterns(2, :), -scale*U(1, :, 1), -scale*U(2, :, 1), 'AutoScale', 'off' ) ;
                set(l_J_2,'LineWidth',2,'Color', green, 'ShowArrowHead', 'off')
                % arrows in direction smallest component
                l_j_1 = quiver( obj.patterns(1, :), obj.patterns(2, :), scale*U(1, :, 2), scale*U(2, :, 2), 'AutoScale', 'off' ) ;
                set(l_j_1,'LineWidth',2,'Color', red, 'ShowArrowHead', 'off')
                l_j_2 = quiver( obj.patterns(1, :), obj.patterns(2, :), -scale*U(1, :, 2), -scale*U(2, :, 2), 'AutoScale', 'off' ) ;
                set(l_j_2,'LineWidth',2,'Color', red, 'ShowArrowHead', 'off')

                % plot histogram of singular values
                % plot_singular_values( S(:) ) ;

                % nullclines
                % [~, l_nc1] = contour(x, y, X-f1,[0, 0], 'linewidth', 1, 'color', [0.2, 0.2, 0.2], 'linestyle', '--') ;
                % [~, l_nc2] = contour(x, y, Y-f2,[0, 0], 'linewidth', 1, 'color', [0.2, 0.2, 0.2], 'linestyle', ':') ;

                % patterns to memorize
                l_patterns = plot(obj.patterns(1, :), obj.patterns(2, :), '.', 'MarkerSize', 20, 'color', orange) ;

                % show generated samples
                if nargin >= 4 && ~isempty(varargin{3})
                    samples = varargin{3} ;
                    % plot(samples(1, :), samples(2, :), '-', 'color', green)
                    plot(samples(1, :), samples(2, :), '.', 'MarkerSize', 10, 'color', green)
                end
                
                hold off
                set(gca,'FontSize',12)
                xlabel('$x_1$', 'interpreter', 'latex', 'fontsize', 14)
                ylabel('$x_2$', 'interpreter', 'latex', 'fontsize', 14)
                xlim([-wdw, wdw])
                ylim([-wdw, wdw])
                % xticks([])
                % yticks([])
                % axes through origin
                % axis equal
                % title( obj.name, 'interpreter', 'latex', 'fontsize', 14 )
                % legend( [l_patterns, l_nc1, l_nc2], {'Pattern', '$x_1$ nullcline', '$x_2$ nullcline'}, 'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;
                % title( "Vector field learned by contractive autoencoder", 'interpreter', 'latex', 'fontsize', 14 )
                % legend( [l_patterns, l_man, qui, l_J_2], ...
                %         {'Data point $\mathbf{x}_p$', 'Manifold', 'Vectorfield', "Principal direction of jacobian"}, ...
                %         'location', 'southwest','interpreter', 'latex', 'fontsize', 12) ;

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