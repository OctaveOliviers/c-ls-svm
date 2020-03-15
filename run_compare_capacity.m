% @Author: OctaveOliviers
% @Date:   2020-03-14 19:56:52
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-15 09:35:41

% compare capacity of different feature maps

clear all
clc


max_dim 		= 10 ;
tol 			= 1e-3 ;
num_tests 		= 100 ;

% feature maps to test
spaces 			= {	'dual',	'dual',	'dual',	'dual' } ;
phis 			= {	'tanh',	'sign',	'poly',	'poly' } ;
thetas			= {	0, 			0, 		[3, 1],	[7, 1] } ;

data 			= cell( length(phis), 4 ) ;
[data{:, 1}] 	= deal( spaces ) ;
[data{:, 2}] 	= deal( phis ) ;
[data{:, 3}] 	= deal( thetas ) ;
[data{:, 4:5}] 	= deal( zeros(1, 10) ) ;

cap_hopfld 		= 0.138*1:max_dim ;

for d = 1:max_dim
	for p = 1:length(phis)
		for i = 1:num_tests
			% capacity until recall error is larger than tol
			cap 			= run_one_test( spaces{p}, phis{p}, thetas{p}, d, tol );

			mean_prev		= data{ p, 4 }(d) ;
			% running acerage
			data{ p, 4 }(d) = mean_prev + ( cap - mean_prev )/i ;
			% running variance
			data{ p, 5 }(d) = data{ p, 5 }(d) + ( cap - mean_prev )*( cap - data{ p, 4 }(d) ) ;
		end
	end
end


figure( 'position', [100, 100, 400, 300] )
box on
hold on
plot(1:max_dim, cap_hopfld, 'linestyle', '-', 'linewidth', 1) ;
for p = 1:length(phis)
	errorbar(1:max_dim, data{p, 4}, sqrt(data{p, 5}/num_tests), 'linestyle', '-', 'linewidth', 1) ;
end
hold off
legend( ['Hopfield', phis], 'location', 'best' )
xlabel( 'Network size N' )
ylabel( 'Network maximal capacity C^*' )
title( {'The capacity of the network', 'strongly depends on the feature map'} )



function cap = run_one_test( space, fun, param, dim, tol )
	% fun 		feature map to use for test
	% param		parameter of feature map
	% dim		dimension of network to evaluate
	% tol		tolerance for error

	% cap = randn + 2 ;

	data 		= rand( dim, 1 ) ;
	err_rate 	= 0 ;

	% create model
	p_err  		= 1e4 ;	% importance of error
	p_reg  		= 1e1 ;	% importance of regularization
	p_drv  		= 1e3 ;	% importance of minimizing derivative
	model 		= Memory_Model_Shallow( space, fun, param, p_err, p_drv, p_reg) ;

	while err_rate < tol

		% append new data point
		data( :, end+1 ) = rand( dim, 1 ) ;

		% train model
		model = model.train( data ) ;

		% simulate


		% estimate error rate

		% avoid being trapped forever
		if size(data, 2)>10*dim
			break
		end

	end
	
end