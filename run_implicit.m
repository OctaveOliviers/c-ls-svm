% Created  by OctaveOliviers
%          on 2020-03-29 17:04:25
%
% Modified on 2020-03-30 20:37:13

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_X   = 1 ;
num_X   = 4 ;
% parameters for training deep model
max_it  = 10 ;
alpha   = 1 ;

% hyper-parameters
p_err   = 1e3 ; % importance of error
p_reg   = 1e-2 ; % importance of regularization
p_drv   = 1e1 ; % importance of minimizing derivative

num_lay = 2;
formul  = { "d", "p" } ;
phis    = { "poly", "tanh" } ;
thetas  = { [1, 0], 0 } ;

% initialize random number generator
rng(10) ;

% create patterns to memorize
X = [ 1, 4, 5 ] ;

% % build models
% level_1 = build_model( 1, 'd', 'poly', [1, 0], p_err, p_drv, p_reg ) ;
% level_2 = build_model( 1, 'p', 'tanh', 0, p_err, p_drv, p_reg ) ;

% % train models
% level_1 = level_1.train( X ) ;
% level_2 = level_2.train( X ) ;

% % update target hidden state
% for i = 1:maxit
%     % old objective function
%     O_old = level_1.lagrangian() + level_2.lagrangian()


%     % extract usefull variables
%     H = level_1.Y ;
%     E1 = level_1.E ;
%     E2 = level_2.E ;
%     J1 = level_1.J ;
%     J2 = level_2.J ;
%     W1 = level_1.W ;
%     W2 = level_2.W ;


%     % derivative of weights to hidden state
%     dW1 = X / ( sum( X.^2 + level_1.p_drv/level_1.p_err*ones(size(X)) ) + level_1.p_reg/level_1.p_err ) ;
 
%     dW2 =   ( (X-level_2.b) .* jac( H, level_2.phi ) ) / ...
%             ( sum( feval(level_2.phi, H).^2 + level_2.p_drv/level_2.p_err*jac( H, level_2.phi ).^2 ) + level_2.p_reg/level_2.p_err ) - ...
%             ( sum( (X-level_2.b).*feval(level_2.phi, H) ) ) * ...
%             ( 2*feval(level_2.phi, H).*jac( H, level_2.phi ) + 2*level_2.p_drv/level_2.p_err*jac( H, level_2.phi ).*squeeze( hes( H, level_2.phi ) ) ) / ...
%             ( sum( feval(level_2.phi, H).^2 + level_2.p_drv/level_2.p_err*jac( H, level_2.phi ).^2 ) + level_2.p_reg/level_2.p_err ).^2 ;

%     % derivatives of lagrangians
%     dL1 =   level_1.p_err * ( E1 - sum( E1.*X ) * dW1 ) + ...    % error term
%             level_1.p_drv * sum( J1 ) * (-dW1) + ...             % derivative term
%             level_1.p_reg * dW1 ;                                % regularization term

%     dL2 =   level_2.p_err * ( - E2 .* W2 .* jac( H , level_2.phi ) - sum( E2 .* feval(level_2.phi, H) ) .* dW2 ) + ...          % error term
%             level_2.p_drv * ( - J2 .* W2 .* squeeze(hes( H , level_2.phi )) - sum( J2 .* jac(H, level_2.phi) ) .* dW2 ) + ...   % derivative term
%             level_2.p_reg * dW2 ;                                                                                               % regularization term

%     % descent direction
%     stp = dL1 + dL2 ;

%     % backtracking
%     b = 1 ;
    
%     % for k = 1:10

%     %     % new objective function
%     %     O_new = level_1.lagrangian( X , H - b*stp ) + ...   % lagrangian at prev level
%     %             level_2.lagrangian( H - b*stp , X ) ;       % lagrangian at current level

%     %     if ( O_new > O_old )
%     %         b = b/2 ;
%     %         disp( "b divided" )
%     %     else
%     %         disp( "backtracking with b = " + num2str(b) )
%     %         break
%     %     end
%     % end
%     % b

%     % level_1.lagrangian(X, H) + level_2.lagrangian(H, X)
%     H = H - b * stp ;

%     O_new = level_1.lagrangian(X, H) + level_2.lagrangian(H, X)

%     % retrain
%     level_1 = level_1.train( X, H ) ;
%     level_2 = level_2.train( H, X ) ;

%     model = Memory_Model_Deep( {level_1, level_2} ) ;
%     model.visualize() ;
%     pause(2)
% end

% visualize model
% model = Memory_Model_Deep( {level_1, level_2} ) ;
%
model = build_model( num_lay, formul, phis, thetas, p_err, p_drv, p_reg, alpha, max_it ) ;

model = model.train_implicit( X ) ;
model.visualize() ;

% % compute error
% E = model1.error( X ) ;

% % train model
% model2 = model1.train( X-E, X ) ;
% % visualize model
% model2.visualize() ;


% % visualize Lagrangian as a fuction of H
% [X, Y] = meshgrid (-5:1:5) ;
% X_vec = X(:) ;
% Y_vec = Y(:) ;
% L = zeros(1, size(X(:), 1)) ;
% for i = size(X_vec, 1)
%     for j = size(Y_vec, 1)
%         L() ;
%     end
% end