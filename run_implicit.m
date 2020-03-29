% @Author: OctaveOliviers
% @Date:   2020-03-28 12:26:03
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-29 12:50:09

clear all
clc

% import dependencies
addpath( './models/' )
addpath( './support/' )

% parameters of patterns
dim_X 	= 1 ;
num_X 	= 4 ;
maxit	= 3 ;

% hyper-parameters
p_err  	= 1e4 ;	% importance of error
p_reg  	= 1e0 ;	% importance of regularization
p_drv  	= 1e2 ;	% importance of minimizing derivative

% initialize random number generator
rng(10) ;

% create patterns to memorize
X = [ 1, 3 ] ;

% build models
level_1 = build_model( 1, 'd', 'poly', [1, 0], p_err, p_drv, p_reg ) ;
level_2 = build_model( 1, 'p', 'sign', 0, p_err, p_drv, p_reg ) ;

% train models
level_1 = level_1.train( X ) ;
level_2 = level_2.train( X ) ;

% update target hidden state
for i = 1:maxit
	level_1.lagrangian()
	level_2.lagrangian()

	H = level_1.Y ;
	E1 = level_1.E ;
	E = level_2.E ;
	J = level_2.J ;
	w = level_2.W ;
	% descent direction
	stp = level_1.p_err*E1 + level_2.p_err*E.*J + level_2.p_drv*J.*2.*w.*tanh(H).*(1-tanh(H).^2) ;

	% backtracking
	b = 1 ;
	% old objective function
	% O_old = level_1.lagrangian() + level_2.lagrangian() ;
	% for k = 1:10

	% 	% new objective function
	% 	O_new = level_1.lagrangian( X , H - b*stp ) + ... 	% lagrangian at prev level
	% 			level_2.lagrangian( H - b*stp , X ) ;		% lagrangian at current level

	% 	if ( O_new > O_old )
	% 		b = b/2 ;
	% 		disp( "b divided" )
	% 	else
	% 		disp( "backtracking with b = " + num2str(b) )
	% 		break
	% 	end
	% end
	% b
	H = H - b * stp ;

	% retrain
	level_1 = level_1.train( X, H ) ;
	level_2 = level_2.train( H, X ) ;

	% model = Memory_Model_Deep( {level_1, level_2} ) ;
	% model.visualize() ;
	% pause(2)
end

% visualize model
model = Memory_Model_Deep( {level_1, level_2} ) ;
model.visualize() ;

% % compute error
% E = model1.error( X ) ;

% % train model
% model2 = model1.train( X-E, X ) ;
% % visualize model
% model2.visualize() ;


% visualize lagrangian of second model
[X, Y] = meshgrid( -5:1:5 ) ;
