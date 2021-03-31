x = [2, 7, -7, 1] ;

% (hyper-)parameters of the layer
space           = 'dual' ;
dim_input       = size(x,1) ;
hp_equi         = 1e2 ;
hp_stab         = 1e2 ;
hp_reg          = 1e-2 ;
feat_map        = 'rbf' ;
feat_map_param  = 8 ;

% define the model
model = CLSSVM( ) ;

% add a layer
model = model.add_layer( space, dim_input, hp_equi, hp_stab, hp_reg, feat_map, feat_map_param ) ;