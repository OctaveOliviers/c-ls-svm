% Created  by OctaveOliviers
%          on 2020-03-29 19:28:02
%
% Modified on 2020-03-29 19:32:59

classdef Memory_Model_Deep < Memory_Model
    
    properties
        models      % cell of shallow models in each layer
        num_lay     % number of layers
        max_iter    % maximum number of iterations during training
        alpha       % learning rate for gradient descent in hidden states
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
                obj.max_iter    = 20 ;
                obj.alpha       = 0.01 ;
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
                obj.max_iter    = 20 ;
                obj.alpha       = 1 ;
                % shallow model for each layer
                for l = 1:obj.num_lay
                    obj.models{l} = build_model(1, varargin{2}{l}, varargin{3}{l}, varargin{4}{l}, varargin{5}, varargin{6}, varargin{7}) ;
                end
            end
        end


        % train model to implicitely find good hidden states with target propagation
        function obj = train_implicit( obj, X, varargin )
            % X         patterns to memorize
            % varargin  contains Y to map patterns X to (for stacked architectures)
            
            % extract useful parameters
            [N, P]          = size(X) ;
            obj.X           = repmat( X, 1, 1, obj.num_lay+1 ) ;

            for i = 1:obj.max_iter
                % hidden representations of patterns
                H = obj.X ;

                % train each layer
                for l = 1:obj.num_lay
                    obj.models{ l } = obj.models{ l }.train( H(:, :, l), H(:, :, l+1) ) ;
                end

                % evaluate objective value
                L = obj.lagrangian()

                % update hidden layers
                for l = obj.num_lay-1:-1:1

                    assert( strcmp(obj.models{l}.phi, 'sign'), 'deep target only for sign(x) yet' )

                    switch obj.models{ l }.space
                        case {'primal', 'p'}
                            L_e_l   = obj.models{ l }.L_e ; 
                            % F_lp1 = jac( H(:, :, l), obj.phi{l+1}, obj.theta{l+1} ) ;
                            % W_lp1 = obj.models{ l+1 }.W ;
                            % L_e_lp1   = obj.models{ l+1 }.L_e ;
                            % L_d_lp1   = obj.models{ l+1 }.L_d ;

                            % grad  = L_e_l - F_lp1'*W_lp1*L_e_lp1 ;
                            % for p = 1:P
                            %   A = L_d_lp1(:, (p-1)*N+1:p*N)' * W_lp1 ;

                            %   H = hess() ;
                            % end

                            grad = L_e_l ;
                            r = max(vecnorm(grad))

                            H(:, :, l+1) = H(:, :, l+1) - obj.alpha * grad ;

                        case {'dual', 'd'}
                            warning( 'target prop has not yet been implemented for dual formulation' ) ;
                    end

                end
                obj.X = H ;


                % check for convergence
                if ( r < 1e-5 )
                    break
                end

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