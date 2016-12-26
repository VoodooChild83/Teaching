function [ Param, StdErr, Avar, J2, J1, h2, h1 ] = GMM( guess,Y,X,Z,moment,momentc,e,q )

% This function will calculate the efficient 2-step GMM estimate

% The moment equation (defined as an anonymous function within the main
% script) should be provided as an input into this function for GMM. 

warning('off','all');

options = optimoptions(@fminunc,'TolFun',1e-20,'TolX',1e-20,'MaxFunEvals',15000,'MaxIter',20000,'Display','off');

%% 1st step: Consistent estimate of parameter

W=inv(Z'*Z); %eye(size(Z,2));

[Param,J1]=fminunc(@(b) J(b,moment,Y,X,Z,W),guess,options);

%% Generate the Efficient Weight Matrix W:

if nargin>7
[~,Omegahat,h1]=HAC(Param,Y,X,Z,momentc,e,q);
else 
[~,Omegahat,h1]=HAC(Param,Y,X,Z,momentc,e);
end

W=inv(Omegahat);
%% 2nd Step: Estimate parameter using the optimal weight amtrix

[Param,J2]=fminunc(@(b) J(b,moment,Y,X,Z,W),Param,options);

if nargin>7
[Avar,~,h2]=HAC(Param,Y,X,Z,momentc,e,q);
else
[Avar,~,h2]=HAC(Param,Y,X,Z,momentc,e);
end

StdErr=sqrt(diag(Avar));


end

