function [ S2r1,S2r2,CV ] = Granger( data,p,confidence,flag )
%% DOCSTRING
%
% This function will conduct a granger causality in the bivariate case. In
% the restricted model the off diagnal elements of the PI array for the
% direction of causality are set to 0.
%
% INPUT:
% data - a Tx(K+1) dataset (K+1 since the first column is assumed to be a
% timestamp --> The first column must be either a timestamp or NaN)
% p - the number of lags for the model
% confidence - a number in (0,1) that designates the confidence level for
% the F critical value
% flag - Model is to include a constant (flag==1) or no constant (flag==0).
%
% OUTPUT:
% S2r1 - The test statistic for the direction Y2t -> Y1t (Y2 granger causes
% Y1?)
% S2r2 - The test statistic for the direction Y1t -> Y2t (Y1 granger causes
% Y2?)
% CV - The critical value of the F distribution under the T-2*p-1 deg of
% freedom
%
% FUNCTIONS USED:
% VAR.m

%% Prep variables

[T,K]=size(data(:,2:end));          % T=# of observations; K=# of variables

%% Estimate the unrestricted model 

[cons,PI_u,~,ehat1,X,Y]=VAR(data,p,flag);

% The Sum of Squared Residuals from the unrestricted model:

SSR_u=sum(diag(ehat1'*ehat1)); % RSS of unrestricted model

%% Estimate the Restricted model: Direction Y2t -> Y1t

% Create a copy of the unrestricted parameter aray
PI_r = PI_u;

% Set restriction: A12=0 for all A block arrays within PI
PI_r(1,K:K:end) = 0; 

% Residuals of the estimation
e1_r = Y - X*[cons PI_r]';   

% Compute restricted SSR
SSR_r = sum(diag(e1_r'*e1_r));    

% Compute the test statistic
S2r1 = ((SSR_r-SSR_u)/p)/((SSR_u)/(T-2*p-1));

%% Estimate the Restricted model: Direction Y1t -> Y2t

% Create a copy of the unrestricted parameter aray
PI_r = PI_u;

% Set restriction: A12=0 for all A block arrays within PI
PI_r(2,K-1:K:end) = 0; 

% Residuals of the estimation
e1_r = Y - X*[cons PI_r]';   

% Compute restricted SSR
SSR_r = sum(diag(e1_r'*e1_r));    

% Compute the test statistic
S2r2 = ((SSR_r-SSR_u)/p)/((SSR_u)/(T-2*p-1));

%% Critical Value of the F distribution

CV = icdf('F',confidence,p,T-2*p-1);
end

