function [ Avar,Omegahat,p ] = HAC( param,Y,X,Z,q)
% This function will calculate the robust variance-covariance matrix for
% heteroskedasticity and autocorrelation.
% Note that for OLS correction we would pass in the X matrix for the Z
% matrix.

% If the argument does not pass in a bandwidth, q, then it will simply
% calculate the hetertoskedastically robust variance-covariance matrix.
% Prior to estimating, this function will conduct a Ljung-Box Q test for
% autocorrelation/serial correlation (as the case may be) with a lag
% structure up to ln(sample size) - to prevent a decrease in statistical
% power at large lags) - and at the default 95% confidence level. 

N=length(Y);

%% Generate the estimated errors and SSR:

ehat=Y-X*param;
SSR=ehat.*ehat;

B=diag(SSR);

%% Ljung-Box Q Test for Autocorrelation of Error Terms

% Test for autocorrelation/serial correlation of the estimated residuals at
% the defaul 95% confidence level.

[~,p]=lbqtest(ehat,'lags',floor(log(N)));
   
%% The Omega hat matrix
   
Omegahat=(Z'*B*Z);

%% Now correct for HAC variance-covariance matrix

% The below will implement HAC if two conditions are satisfied. 
% 1) if bandwidth q is included but LBQ test fails to reject the null (no
% autocorrelation), then HAC will not be generated.

if nargin==5 && p<0.05 %|| nargin==4 && p<0.05;
    
%     if nargin==4
%        
%        q=floor(0.75*N^(1/3));
%        
%     end
    
    % The Newey-West correction for the autocorrelation can be obtained
    % from the below:
    for j=1:q
        Gq=zeros(rows(Omegahat),cols(Omegahat));
        for k=j+1:N
            Gq=Gq+ehat(k)*ehat(k-j)*Z(k,:)'*Z(k-j,:);
        end
        Omegahat=Omegahat+(1-j/(q+1))*(Gq+Gq');
    end
    
    % The HAC variance-covariance matrix:
    Avar=inv((X'*Z)/Omegahat*(Z'*X));   
    
else
    
    % The heteroskedastically robust variance-covariance matrix:
    Avar=inv((X'*Z)/Omegahat*(Z'*X));
    
end

end

