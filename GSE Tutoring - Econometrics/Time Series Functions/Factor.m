function [ f,Eigen ] = Factor( data, p )
%% DOCSTRING
%
% The function generates the factor array based on the data provided
%
% INPUT:
% data - a Tx(K+1) dataset (K+1 since the first column is assumed to be a
% timestamp --> The first column must be either a timestamp or NaN)
%
% p - the number of lags to include in the model (optimal lags assumed to
% already have been conducted outside of the function).
%
% OUTPUT:
% f - Txf array of the panel of factors
%
% Eigen - the eigen values of the VarCov array

%% Prepare Data

% Remove the date-stamp column
data = data(:,2:end);

% Use cover function to generate the centered varcov array
DataCov = cov(data); 


%% Generate the factors

% Obtain the eigen vectors and eigen values of the varcov array
[EigV,Eigen] = eig(DataCov);

% Rearrange the matrices by rotating them and/or flipping them
% left-to-right (reorder to obtain highest eigen value first)
Eigen = fliplr(rot90(Eigen(end-r+1:end,end-r+1:end)));

EigV = fliplr(EigV(:,end-r+1:end));

% Now generate the factors:
f = (EigV'*data')';

end

