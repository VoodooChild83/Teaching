function [ Avar,Omegahat,h ] = HAC( param,Y,X,Z,momentc,e,q)
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

ehat=e(param,Y,X);
% ehat=Y-X*param;
% SSR=ehat.*ehat;
% 
% B=diag(SSR);

%% Ljung-Box Q Test for Autocorrelation of Error Terms

% Test for autocorrelation/serial correlation of the estimated residuals at
% the default 95% confidence level. Null (no autocorrelation) rejection =>
% h=1; fail to reject null => h=0.

[h]=lbqtest(ehat,'lags',1:round(log(N)));

h=double(h);    % convert to a double
   
%% The Omega hat matrix
   
% Omegahat=(Z'*B*Z);

MC=momentc(param,Y,X,Z);

Omegahat=MC'*MC;

%% Now correct for HAC variance-covariance matrix

% The below will implement HAC if two conditions are satisfied. 
% 1) if bandwidth q is included but LBQ test fails to reject the null (no
% autocorrelation), then HAC will not be generated.
% 2) if bandwidth is not included but we reject the null in LBQ test (that 
% there is autocorrelation), then the function generates the HAC var-covar
% matrix

if nargin>6 %&& norm(h)>0 %|| nargin<6 && norm(h)>0
    
%     if nargin==4
%        
%        q=round(0.75*N^(1/3));
%        
%     end
    
     % The Newey-West correction for the autocorrelation can be obtained
    % from the below:
    
%% Michael's Code for Newey West

    Zed = MC - repmat(mean(MC,1),N,1);
    
    for i=1:q
        Zedlag=Zed(1:N-i,:);
        ZZ=Zed(1+i:N,:);
        Omegahat=Omegahat+(1-i/(q+1))*((ZZ'*Zedlag)+(ZZ'*Zedlag)');
    end
    
%% My code
    
%     for j=1:q
%         Gq=zeros(rows(Omegahat),cols(Omegahat));
%         for k=j+1:N
%             Gq=Gq+ehat(k)*ehat(k-j)*Z(k,:)'*Z(k-j,:);
%         end
%         Omegahat=Omegahat+(1-j/(q+1))*(Gq+Gq');
%     end
    
%% The HAC variance-covariance matrix:
    Avar=inv((X'*Z)/Omegahat*(Z'*X));   
    
else
  
    % The heteroskedastically robust variance-covariance matrix:
    Avar=inv((X'*Z)/Omegahat*(Z'*X));
    
end

end

