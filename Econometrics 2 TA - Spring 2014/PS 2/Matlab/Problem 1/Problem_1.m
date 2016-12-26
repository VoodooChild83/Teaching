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
% 
% %% Problem Set 2
% 
% % Define the moment conditions:
% 
% %theta=ones(2,1);
% % Based on the Score (we get two moment conditions):
% GMM.m1 = @(theta) variable.X'*variable.y - variable.X'*logisticcdf(-variable.X*theta);
% 
% % Based on the mean:
% GMM.m2 = @(theta) mean(logisticcdf(-variable.X*theta)) - mean(variable.y);
% 
% % Based on the variance:
% GMM.m3 = @(theta) mean(logisticcdf(-variable.X*theta).*(1-logisticcdf(-variable.X*theta))) - var(variable.y);
% 
% % Create the moment matrix:
% GMM.m= @(theta) [GMM.m2(theta);GMM.m3(theta)];
% 
% % First Step of efficient GMM: Define the weight matrix as identity of size
% % of the GMM.m matrix
% 
% GMM.W1=@(theta) eye(size(GMM.m(theta),1));
% 
% % Define the distance function:
% 
% GMM.J1=@(theta) GMM.m(theta)'*GMM.W1(theta)*GMM.m(theta);
% 
% % First pass minimization of the distance function:
% 
% [GMM.theta1] = fminunc(@(theta) GMM.J1(theta),ols.beta,options);
% 
% display(GMM.theta1);
% 
% % Now 2nd step GMM:
% 
% % GMM.M1=GMM.m1(GMM.theta1);
% GMM.M2=GMM.m2(GMM.theta1);
% GMM.M3=GMM.m3(GMM.theta1);
% 
% GMM.M=[GMM.M2;GMM.M3];
% display(GMM.M);
% 
% GMM.W2=(GMM.M*GMM.M')\eye(size(GMM.M,1));
% display(GMM.W2);
% 
% GMM.J2=@(theta) GMM.m(theta)'*GMM.W2*GMM.m(theta);
% 
% [GMM.theta2] = fminunc(@(theta) GMM.J2(theta),GMM.theta1,options);
% 
% display(GMM.theta2);
% 







