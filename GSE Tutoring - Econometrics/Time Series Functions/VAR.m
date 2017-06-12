function [ C,PI,Eigen,ehat,X,Y ] = VAR( data,p,constant )
%% DOCSTRING
%
% The function estimates a VAR based on the SUR representation of the
% model. It will also estimate an AR process if the size of the data array
% is K=1 (univariate estimation).
%
% INPUT:
% data - a Tx(K+1) dataset (K+1 since the first column is assumed to be a
% timestamp --> The first column must be either a timestamp or NaN)
%
% p - the number of lags to include in the model (optimal lags assumed to
% already have been conducted outside of the function).
%
% constant - if the model is to include a constant (constant==1) or no
% constant(==0 or any other number)
%
% OUTPUT:
% C - Kx1 array of the constants of the model
%
% PI - Kx(K*p) array of the model coefficients (block arrays of size KxK
% for p lag length)
%
% Eigen - the eigen values of the system (if needed)
% ehat - TxK residuals of the estimation
%
% X - the design matrix used in the estimation
%
% Y - the dependant values used in the estimation
%
% FUNCTIONS USED:
% Companion.m

%% Declare the Variables from data input: Generate the Matrices

Y = data(p+1:end,2:end);      % The TxK matrix: T = time obs, K = variables
                              % Remove the first p observations

X = lagmatrix(data(:,2:end),1:p);

X(1:p,:)=[];                  % Remove the first p observations              

if constant==1
    X = [ones(size(X,1),1) X];
end

%% Estimate the SUR representation:

beta = ((X'*X)\X'*Y)';       % We have here the Kx(K*p+1) array

%% Extract the Submatrices:

if constant==1
   C = beta(:,1);             % The Kx1 vector of the constant's coefficient
   PI = beta(:,2:end);        % The Kx(K*p) matrix of coefficents, where 
                              % within are the relevant KxK sub matrices that
                              % correspond to the matrices A1, A2, A3,...
                              % for each of the Tx(K*p) lag vectors
else
   C = zeros(size(beta,1),1);   % A constant of zeros Kx1
   PI = beta;
end

%% Generate Eigen Values

A = Companion(PI);

Eigen = eig(A);

%% The Estimated Residuals

ehat = Y - X*beta';          % TxK matrix of the residuals of the estimation

end

