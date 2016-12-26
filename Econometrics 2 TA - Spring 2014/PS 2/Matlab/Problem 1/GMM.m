function [ delta, Avar, Omegahat, J ] = GMM( Y,X,Z,q )
% This function will calculate the GMM estimate through two step GMM
     [N K]=size(X);
% Projection matrix
     P=Z/(Z'*Z)*Z';
% First-step ineff but consistent coeff estimates are IV
     delta=(X'*P*X)\X'*P*Y;
% Residuals
     ehat=Y-X*delta;
% Omega calculation
     SSR=(ehat*ehat');
     B=zeros(N);
for i=1:N
    for j=1:N
        if i==j
           B(i,j)=SSR(i,j);
        end
    end
end
    Omegahat=Z'*B*Z;
% Correct the Autocorrelation
for j=1:q
    Gq=zeros(rows(Omegahat),cols(Omegahat));
    for k=j+1:N
        Gq=Gq+ehat(k)*ehat(k-j)*Z(k,:)'*Z(k-j,:);
    end
    Omegahat=Omegahat+(1-j/(q+1))*(Gq+Gq');
end

% Recalculate delta coefficients
    delta=(X'*Z/Omegahat*Z'*X)\(X'*Z)/Omegahat*Z'*Y;
% And now the assymptotic variance-covariance matrix of the coefficients
% Hetero and Auto Corrected
    Avar=inv(X'*Z/Omegahat*Z'*X);
% J statistic:
    J=(Z'*(Y-X*delta))'/Omegahat*(Z'*(Y-X*delta));
end

