% Created  by OctaveOliviers
%          on 2020-03-29 19:31:12
%
% Modified on 2020-09-26 17:41:10

% classdef Hopfield_Network < CLSSVM
classdef Hopfield_Network

    properties
        X        % memories
        algo     % training algorithm for the weights
        W        % primal weights
        name     % name of model
    end

    methods
        % constructor
        function obj = Hopfield_Network(memories, algo)
            obj.X  = memories ;
            % training algorithm
            obj.algo = algo ;     % string
            % model information
            obj.name    = "Hopfield Network" ;

            obj = obj.train() ;
        end


        % train model
        function obj = train(obj)
            % X         patterns to memorize in columns (assume -1/1 patterns)
            
            % extract useful parameters
            [N, P] = size( obj.X ) ;

            switch lower(obj.algo)

                case {'hebb', 'hebbian', 'h'}
                    obj.W = (obj.X * obj.X') / N ;

                case {'storkey', 's'}
                    error( 'Not implemented yet' )

                case {'pseudo-inverse', 'pseudoinverse', 'pseudo','pi'}
                    obj.W = obj.X * pinv(obj.X) ;

                otherwise
                    error( 'Did not recognize training algorithm. Can be "primal", "p", "dual" or "d".' )
            end

            
            disp("Hopfield model trained")
        end


        % simulate model over one step
        function f = simulate_one_step(obj, x)
            % x     matrix with start positions to simulate from as columns

            f = sign( obj.W' * x ) ;
        end

    end
end