function model = clssvm_rbf(dim_pat)
    space           = 'dual' ;          % space to train layer
    hp_equi         = 1e2 ;             % importance of equilibrium objective
    hp_stab         = 1e2 ;             % importance of local stability objective
    hp_reg          = 1e-2 ;            % importance of regularization
    feat_map        = 'rbf' ;           % chosen feature map or kernel function
    feat_map_param  = 0.5 ;             % parameter of feature map or kernel function
    
    model = CLSSVM( ) ; 
    model = model.add_layer( space, dim_pat, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;
end