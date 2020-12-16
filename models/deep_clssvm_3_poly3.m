function model = deep_clssvm_3_poly3(dim_pat)

    spaces           = {'primal', 'primal', 'primal' } ;
    dim_inputs       = {dim_pat, dim_pat, dim_pat} ;
    hp_equis         = {1e2, 1e2, 1e2} ;
    hp_stabs         = {1e0, 1e0, 1e0} ;
    hp_regs          = {1e-2, 1e-2, 1e-2} ;
    feat_maps        = {'poly', 'poly', 'poly'} ;
    feat_maps_param  = {[3, 1], [3, 1], [3, 1]} ;
    
    model = CLSSVM( ) ;
    
    for l = 1:length(feat_maps)
        model = model.add_layer( spaces{l}, dim_inputs{l}, hp_equis{l}, hp_stabs{l}, hp_regs{l}, feat_maps{l}, feat_maps_param{l} ) ;
    end
end