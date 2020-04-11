% @Author: OctaveOliviers
% @Date:   2020-03-15 16:54:15
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 18:31:49

function model = build_model( num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg )

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
		model = Memory_Model_Deep(num_layers, formulation, feature_map, parameter, p_err, p_drv, p_reg) ;
	end
end