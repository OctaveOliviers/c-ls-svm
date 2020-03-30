% Created  by OctaveOliviers
%          on 2020-03-28 15:14:14
%
% Modified on 2020-03-30 17:44:25

function model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg, varargin )

    if ( num_layers ==1 )
        
        switch formulation
            case { 'primal', 'p' }
                model = Memory_Model_Shallow_Primal( feature_map, parameter, p_err, p_drv, p_reg ) ;

            case { 'dual', 'd' }
                model = Memory_Model_Shallow_Dual( feature_map, parameter, p_err, p_drv, p_reg ) ;

            otherwise
                error( 'Did not recognize "space" variable. Can be "primal", "p", "dual" or "d".' )
        end

    else
        model = Memory_Model_Deep( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg, varargin{:} ) ;
    end
end