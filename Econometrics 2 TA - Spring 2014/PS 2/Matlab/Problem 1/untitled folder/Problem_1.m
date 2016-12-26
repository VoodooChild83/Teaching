% Alberto Ramirez
% logit Problem, Problem 1 in PS1 and PS2
%
% This is the GMM estimation continuation of the PS1 problem now using the
% moment conditions.
% 
%% Problem Set 1

clc
clear all
%
% Define the variables and the data set:
variable.n=10;
variable.y=[0 0 0 1 1 1 1 1 1 1]';          % nx1 vector
variable.x=[-1 -1 1 0 -1 -1 1 1 2 2]';      % nx1 vector
variable.X=[ones(variable.n,1) variable.x]; % Data matrix

variable.Z=[variable.X variable.x.^2];

options = optimoptions(@fminunc,'TolFun',1e-15,'TolX',1e-15,'MaxFunEvals',15000,'MaxFunEvals',20000);

% Estimate first using OLS to create estimate of parameters. Here I will
% allow for a randomization of the OLS beta so that initial guesses are not
% always the same:

ols.beta=(variable.X'*variable.X)\variable.X'*variable.y;

% Now I will call my function to maximize the liklihood function. In last
% semester's class I created a function called 'MLE_Logit' which I will
% call upon again for this assignment. Within this function I had to define
% the logistic CDF function to do the work.

% Per the requirement I will maximize utilizing fminunc:

[L.theta]=fminunc(@(l) MLE_Logit(l,variable.X,variable.y),ols.beta);

fprintf('\n The ML Estimators for the Logit model are: theta1 = %.4f and theta2 = %.4f. \n \n', L.theta(1),L.theta(2));

%% Problem Set 2

% Define the moment conditions:

Moment.e=@(m,Y,X) Y-exp(X*m).*logisticcdf(X*m);

Moment.Moment = @(m,Y,X,Z) Z'*Moment.e(m,Y,X);

Moment.MC = @(m,Y,X,Z) diag(Moment.e(m,Y,X))*Z;

% Estimate with the codes:

[EGMM.delta,EGMM.StdErr]=GMM(ols.beta,variable.y,variable.X,variable.Z,Moment.Moment,Moment.MC,Moment.e);

% EGMM.thetaboot=zeros(2,variable.Reps);
% 
% for i=1:variable.Reps
%     
%     variable.index=randi(variable.n,variable.n,1);
%     
%     variable.Xboot=variable.X(variable.index,:);
%     variable.Yboot=variable.Y(variable.index);
%     variable.Zboot=variable.Z(variable.index,:);
%     
%     EGMM.thetaboot(:,i)=GMM(ols.theta,variable.Yboot,variable.Xboot,variable.Zboot,Moment.Moment,Moment.MC,Moment.e);
%     
%     EGMM.lambdaboot(:,i)=exp(variable.Xboot*EGMM.thetaboot(:,i));
%     
% end










