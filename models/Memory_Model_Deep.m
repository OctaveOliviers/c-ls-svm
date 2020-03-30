% Created  by OctaveOliviers
%          on 2020-03-29 19:28:02
%
% Modified on 2020-03-30 20:37:00

classdef Memory_Model_Deep < Memory_Model
    
    properties
        models      % cell of shallow models in each layer
        num_lay     % number of layers
        max_iter    % maximum number of iterations during training
        alpha       % learning rate for gradient descent in hidden states
        max_back_track = 10 ; % maximum number of times to backtrack
    end

    methods
        % constructor
        function obj = Memory_Model_Deep( varargin )
            % construct either from cell of ordered shallow models as
            %   obj = Memory_Model_Deep( models )
            %
            % or from its parameters as
            %   obj = Memory_Model_Deep( num_lay, spaces, phis, thetas, p_err, p_drv, p_reg )

            % superclass constructor
            obj = obj@Memory_Model() ;

            if (nargin==1)
                % subclass specific variables
                obj.num_lay     = length(varargin{1}) ;
                obj.models      = varargin{1} ;
                obj.max_iter    = varargin{end} ;
                obj.alpha       = varargin{end-1} ;
                obj.X           = varargin{1}{1}.X ;
                obj.name        = join([ num2str(obj.num_lay), '-layered network (']) ;
                for l = 1:obj.num_lay
                    obj.name    = append( obj.name, join([obj.models{l}.phi, ', ']) ) ;
                end
                obj.name    = append( obj.name(1:end-2), ')' ) ;

            else
                % check correctness of inputs
                assert( length(varargin{2}) == varargin{1} , 'Number of spaces does not match number of layers' ) ;
                assert( length(varargin{3}) == varargin{1} , 'Number of feature maps does not match number of layers' ) ;
                assert( length(varargin{4}) == varargin{1} , 'Number of feature parameters does not match number of layers' ) ;

                % subclass specific variables
                obj.num_lay     = varargin{1} ;
                obj.models      = cell(varargin{1}, 1) ;
                obj.max_iter    = varargin{end} ;
                obj.alpha       = varargin{end-1} ;
                % shallow model for each layer
                for l = 1:obj.num_lay
                    obj.models{l} = build_model(1, varargin{2}{l}, varargin{3}{l}, varargin{4}{l}, varargin{5}, varargin{6}, varargin{7}) ;
                end
            end
        end


        % train model to implicitely find good hidden states with target propagation
        function obj = train_implicit( obj, X, varargin )
            % X         patterns to memorize
            
            % extract useful parameters
            [N, P] = size(X) ;
            obj.X  = X ;

            % initialize hidden representations of patterns
            H      = repmat( X, 1, 1, obj.num_lay+1 ) ;
            step   = zeros(size(H)) ;
            % initialize network by training each layer
            for l = 1:obj.num_lay
                obj.models{ l } = obj.models{ l }.train( H(:, :, l), H(:, :, l+1) ) ;
            end

            % train the network
            for i = 1:obj.max_iter
                
                % store current error on patterns
                obj.L = obj.lagrangian() ;
                obj.L = obj.error(X) ;

                % update hidden representations
                for l = obj.num_lay-1:-1:1

                    switch obj.models{l+1}.space

                        case {'primal', 'p'}

                            % extract useful parameters
                            E_l     = obj.models{l}.E ;
                            E_lp1   = obj.models{l+1}.E ;
                            J_lp1   = obj.models{l+1}.J ;
                            W_lp1   = obj.models{l+1}.W ;

                            % derivative at current level
                            dL_l    = obj.models{l}.p_err * E_l ;

                            Hes     = hes( H(:, :, l+1), obj.models{l+1}.phi ) ;

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
                            dL_lp1   = obj.models{l+1}.p_err * dE_lp1 + obj.models{l}.p_drv * dJ_lp1 ;

                            % store update of hidden states
                            step(:, :, l+1) = dL_l + dL_lp1 ;
                            
                        case {'dual', 'd'}
                            warning( 'target prop is not yet implemented for dual formulation' ) ;
                    end
                end

                % backtracking
                b = obj.alpha ;
                for k = 1:obj.max_back_track
                    % store canditate new hidden states
                    H_c = H - b * step ;

                    % train each layer
                    for l = 1:obj.num_lay
                        obj.models{ l } = obj.models{ l }.train( H_c(:, :, l), H_c(:, :, l+1) ) ;
                    end

                    %if ( obj.lagrangian() > obj.L )
                    if ( norm(obj.error(X)) > norm(obj.E) )
                        b = b/2 ;
                    else
                        break
                    end
                end
                b
                if ( k==obj.max_back_track )
                    % did not find better hidden states
                    for l = 1:obj.num_lay
                        obj.models{ l } = obj.models{ l }.train( H(:, :, l), H(:, :, l+1) ) ;
                    end
                    % stop training
                    break
                else
                    disp( "backtracking with b = " + num2str(b) )
                    H = H_c ;
                end

                obj.visualize() ;
                pause(2)

            end

            disp("model trained implicitly")
        end



        % train model to implicitely find good hidden states with target propagation
        function obj = train_explicit( obj, X, H )
            % X         patterns to memorize
            % H         cell containing hidden states
            
            % check correctness of input
            assert( size(H, 2)==(obj.num_lay-1), 'Passed too many hidden states. Should be one less than number of layers.' ) ;

            % extract useful parameters
            [N, P]  = size(X) ;
            obj.X   = X ;

            train_set = [ X, H, X ] ;
            for l = 1:obj.num_lay
                obj.models{l} = obj.models{l}.train( train_set{l}, train_set{l+1} ) ;
            end

            disp("model trained explicitly")
        end


        % compute value of Lagrangian
        function L = lagrangian( obj, varargin )
            L = 0 ;
            for l = 1:obj.num_lay
                L = L + obj.models{ l }.lagrangian( varargin{:} ) ;
            end
        end


        % simulate model over one step
        function F = simulate_one_step( obj, X )
            % x     matrix with start positions to simulate from as columns

            F = X ;
            for l = 1:obj.num_lay
                F = obj.models{ l }.simulate_one_step( F ) ;
            end
        end
    end
end