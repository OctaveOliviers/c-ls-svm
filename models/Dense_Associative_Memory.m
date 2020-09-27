% Created  by OctaveOliviers
%          on 2020-09-24 14:08:34
%
% Modified on 2020-09-26 15:34:14

classdef Dense_Associative_Memory

    properties
        X        % memories
        afun     % activation function
        param    % parameters of activation function
        name     % name of model
    end

    methods
        % constructor
        function obj = Dense_Associative_Memory( memories, afun, param )

            obj.X       = memories ;
            obj.afun    = afun ;
            obj.param   = param ;

            obj.name    = "Dense Associative Memory" ;
        end


        % asynchronously update the state
        function x_kpo = simulate_one_step(obj, x_k)
            % x_k     vector of current state to update

            % initialize updated state
            x_kpo = x_k ;

            % random index to update
            idx = randi( [1, length(x_k)] ) ;
            x_i = x_k(idx) ;

            % energy of 
            sum1 = x_i * sum( feval(obj.afun, obj.param, obj.X' * x_k ) ) ;

            sum2 = -x_i * sum( feval(obj.afun, obj.param, obj.X' * [x_k(1:idx-1) ; -x_i ; x_k(idx+1:end)] ) ) ;

            x_kpo(idx) = sign( sum1 + sum2 ) ;
        end

    end
end