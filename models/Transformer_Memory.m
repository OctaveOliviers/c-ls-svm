% Created  by OctaveOliviers
%          on 2020-09-25 17:15:56
%
% Modified on 2020-09-26 13:07:30

classdef Transformer_Memory

    properties
        X        % memories
        param    % temperature of softmax function
        name     % name of model
    end

    methods
        % constructor
        function obj = Transformer_Memory( memories, param )

            obj.X       = memories ;
            obj.param   = param ;

            obj.name    = "Transformer Memory" ;
        end


        % asynchronously update the model
        function x_kpo = simulate_one_step(obj, x_k)
            % x_k     vector of current state to update

            % initialize updated state
            x_kpo = obj.X * softmax( obj.param * obj.X' * x_k ) ;

        end

    end
end