% @Author: OctaveOliviers
% @Date:   2020-03-20 11:20:11
% @Last Modified by:   OctaveOliviers
% @Last Modified time: 2020-03-20 12:14:17

% adapted from LS-SVMlab1.8 - https://www.esat.kuleuven.be/sista/lssvmlab/

function [eigval, eigvec] = kpca(Xtrain, kernel_type, kernel_pars, etype, nb, rescaling)
% Kernel Principal Component Analysis (KPCA)
%
% >> [eigval, eigvec] = kpca(X, kernel_fct, sig2)
% >> [eigval, eigvec, scores] = kpca(X, kernel_fct, sig2, Xt)
%
% Compute the nb largest eigenvalues and the corresponding rescaled
% eigenvectors corresponding with the principal components in the
% feature space of the centered kernel matrix. To calculate the
% eigenvalue decomposition of this N x N matrix, Matlab's
% eig is called by default. The decomposition can also be
% approximated by Matlab ('eigs'). In some cases one wants to disable
% ('original') the rescaling of the principal components in feature
% space to unit length.
%
% The scores of a test set Xt on the principal components is computed by the call:
%
% >> [eigval, eigvec, scores] = kpca(X, kernel_fct, sig2, Xt)
%
% Outputs
%   eigval       : N (nb) x 1 vector with eigenvalues values
%   eigvec       : N x N (N x nb) matrix with the principal directions
%   scores(*)    : Nt x nb matrix with the scores of the test data (or [])
%   omega(*)     : N x N centered kernel matrix
%   recErrors(*) : Nt x 1 vector with the reconstruction error of the test data
%   optOut(*)    : Optional cell array containing the centered test kernel matrix in optOut{1}
%                  and the squared 2-norm of the test points in the feature space in optOut{2}
% Inputs
%   X            : matrix with the inputs of the training data in columns
%   kernel       : Kernel type (e.g. 'RBF_kernel')
%   sig2         : Kernel parameter(s) (for linear kernel, use [])
%   Xt(*)        : Nt x d matrix with the inputs of the test data (or [])
%   etype(*)     : 'svd', 'eig'(*),'eigs','eign'
%   nb(*)        : Number of eigenvalues/eigenvectors used in the eigenvalue decomposition approximation
%   rescaling(*) : 'original size' ('o') or 'rescaled'(*) ('r')

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab


%
% defaults
%
nb_data = size(Xtrain,2);


if ~exist('nb','var')
    nb=10;
end;

if ~exist('etype','var')
    etype='eig';
end;

if ~exist('rescaling','var')
    rescaling='r';
end;


%
% tests
%
if exist('Xt','var') && ~isempty(Xt) && size(Xt,1)~=size(Xtrain,1),
    error('Training points and test points need to have the same dimension');
end

if ~(strcmpi(etype,'eig') || strcmpi(etype,'eigs')),
    error('Eigenvalue decomposition via ''eig'' or ''eigs''');
end


% kernel matrix
omega = phiTphi( Xtrain, Xtrain, kernel_type, kernel_pars) ;

% Centering
Meanvec = mean(omega,2);
MM = mean(Meanvec);
omega=omega-Meanvec*ones(1,nb_data)-ones(nb_data,1)*Meanvec'+MM;

% numerical stability issues
%omega = (omega+omega')./2;

if strcmpi(etype,'eig'),
    [eigvec, eigval] = eig(omega);
elseif (strcmpi(etype,'eigs')),
    [eigvec, eigval] = eigs(omega,nb);
end
eigval=diag(eigval);

%% Eigenvalue/vector sorting in descending order

[eigval,evidx]=sort(eigval,'descend');
eigvec=eigvec(:,evidx);


%
% only keep relevant eigvals & eigvec
%
peff = find(eigval>1000*eps);
neff = length(peff);

% rescaling the eigenvectors
if (rescaling(1) =='r' && nargout>1)
    for i=1:neff,
        eigvec(:,i) = eigvec(:,i)./sqrt(eigval(i));
    end
end