% Alberto Ramirez
% Poisson Problem, Problem 1 in PS1
%
%
clc
clear all
%
%% Define the variables and the data set:
variable.n=10;

variable.x=[-1 -1 1 0 -1 -1 1 1 2 2]';                % nx1 vector
variable.X=[ones(variable.n,1) variable.x];           % Design matrix

variable.Z=[variable.X variable.x.^2];

variable.Y=[0 0 0 1 1 1 2 2 2 3]';                    % for Poisson

variable.Reps=1000;

%% OLS Guess

% Estimate first using OLS to create estimate of parameters. Here I will
% allow for a randomization of the OLS beta so that initial guesses are not
% always the same:

ols.theta=(variable.X'*variable.X)\variable.X'*variable.Y;

%% Poisson Regression: Structural Form

% I estimate the Poisson model, where lambda has been parameterized by X 
% and theta (I use the OLS beta as a starting guess). 

% Since in this context we are imposing a "structure" on lambda, we do not
% estimate it directly....doing so would be a reduced form estimation of
% the exercise (see next section).

% I will use fminunc. I will also specify that the solver should use a specific
% algorithm by turning on the Gradient (that is, we will be supplying the
% gradient in the objective function for matlab to use in the
% maximization).

% By turning on the gradient, matlab uses the default algorithm
% "trust-region" to maximize the function. This is completely unnecessary
% here, but when objective functions are sensitive to starting values,
% porviding an analytic (or numerical) gradient for matlab to use helps the
% solvers reach global extrema (see below for the example of sensitive
% starting values).

warning('off','all');

Str.options=optimoptions('fminunc','display','off');%,'GradObj','on');

% Run a strucutral estimation of the model:

[Str.theta,~,~,~,~,Str.Hess]=fminunc(@(l) MLE_Poisson(l,variable.X,variable.Y),ols.theta,Str.options);

Str.StdErr=diag(sqrt(inv(Str.Hess)));

% Now calculate the parameter Lambda based on the structural estimates and 
% take the mean:

Str.lambda=mean(exp(-variable.X*Str.theta));

%% Bootstrap

% Bootstrap the standard errors and the point estimates since we are doing
% MLE with small samples, and MLE is an asymptotic theory.

% Str.thetaboot=zeros(2,variable.Reps);
% 
% for i=1:variable.Reps
%     
%     variable.index=randi(variable.n,variable.n,1);
%     
%     variable.Xboot=variable.X(variable.index,:);
%     variable.Yboot=variable.Y(variable.index);
%     
%     Str.thetaboot(:,i)=fminunc(@(l) MLE_Poisson(l,variable.Xboot,variable.Yboot),ols.theta,Str.options);
%     
%     Str.lambdaboot(:,i)=exp(-variable.Xboot*Str.thetaboot(:,i));
%     
% end
% 
% Str.thetabootmean=mean(Str.thetaboot,2);
% Str.thetabootstderr=std(Str.thetaboot,0,2);

% Str.lambdabootmean=mean(mean(Str.lambdaboot));
% Str.lambdabootstderr=std(mean(Str.lambdaboot,1),0,2);

%% Poisson Regression: Reduced Form

RF.lambda1guess=randn(1,1);

% 2 options: Fmincon and Fminunc

% Option 1: Fmincon

% Here we use fmincon since our lambda solution must be positive. The
% parametrization of lambda in the above "structural" exercise took care of
% this and made lambda unconstrained. Since I can generate the gradient
% analytically from the FOC, I also provide that as an output of my
% function and impose that Matlab use the better algorithm to maximize the
% function. Otherwise the solver will be unable to max the function as it 
% is sensitive to the starting value. 

% By turning on the gradient, matlab uses the default algorithm
% "trust-region refelctive", which is a large-scale algorithm that is
% better equipped to find global extrema. In this way I can desensitize the 
% solver to starting values, and our starting lambda may be any real number, 
% even though lambda is defined in the positive portion of the reals.

RF.options=optimoptions('fmincon','GradObj','on','display','off');

[RF.lambda1,~,~,~,~,~,RF.Hess]=fmincon(@(l) MLE_Poisson(l,variable.X,variable.Y),RF.lambda1guess,[],[],[],[],0,Inf,[],RF.options);

RF.StdErr1=sqrt(inv(RF.Hess));
RF.StdErr1=RF.StdErr1(1,1);

% Option 2: Fminunc

% If we want to use fminunc then obviously where we pick our starting point 
% does matter, since any negative starting value will lead to an
% undefined initial starting point, given the form of the poisson
% likelihood function. So instead of picking from the random normal as an
% initial guess, we could use any other distribution that would give us a
% positive starting value (uniform, F, Chi-sq, exponential, etc.)

RF.lambda2guess=exprnd(1,1);

[RF.lambda2,~,~,~,~,RF.Hess2]=fminunc(@(l) MLE_Poisson(l,variable.X,variable.Y),RF.lambda2guess,Str.options);

RF.StdErr2=sqrt(inv(RF.Hess2));

%% Bootstrap Option 1:

% RF.lambdaboot=zeros(variable.Reps,1);
% 
% for i=1:variable.Reps
%     
%     variable.index=randi(variable.n,variable.n,1);
%     
%     variable.Xboot=variable.X(variable.index,:);
%     variable.Yboot=variable.Y(variable.index);
%     
%     RF.lambdaboot(i)=fmincon(@(l) MLE_Poisson(l,variable.Xboot,variable.Yboot),RF.lambda1guess,[],[],[],[],0,Inf,[],RF.options);
%     
% end
% 
% RF.lambdabootmean=mean(RF.lambdaboot);
% RF.lambdabootstderr=std(RF.lambdaboot);

%% Problem Set 2: GMM Estimation

% Define the moment conditions:

Moment.e = @(m,Y,X) Y-exp(X*m);

Moment.Moment = @(m,Y,X,Z) Z'*Moment.e(m,Y,X);

Moment.MC = @(m,Y,X,Z) diag(Moment.e(m,Y,X))*Z;

% Estimate with the codes:

[EGMM.delta,EGMM.StdErr]=GMM(ols.theta,variable.Y,variable.X,variable.Z,Moment.Moment,Moment.MC,Moment.e);

%% Bootstrap the above for confidence intervals

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
%     if size(unique([variable.Yboot,variable.Xboot],'rows'),1)>size(variable.X,2)+1
%     
%        EGMM.thetaboot(:,i)=GMM(ols.theta,variable.Yboot,variable.Xboot,variable.Zboot,Moment.Moment,Moment.MC,Moment.e);
%     
%        EGMM.lambdaboot(:,i)=exp(variable.Xboot*EGMM.thetaboot(:,i));
%        
%     else
%        
%        i=i-1;
%        
%     end
%     
% end
% 
% EGMM.thetabootmean=mean(EGMM.thetaboot,2);
% EGMM.thetabootstderr=std(EGM.thetaboot,0,2);
% 
% EGMM.LB=quantile(EGMM.thetaboot,0.05);
% EGMM.UB=quantile(EGMM.thetaboot,0.95);
% 
% EGMM.lambdabootmean=mean(mean(Str.lambdaboot));
% EGMM.lambdabootstderr=std(mean(Str.lambdaboot,1),0,2);


%% Display results

fprintf('\n The ML Estimators for the Poisson structural model are: theta1 = %.3f (%.3f) and theta2 = %.3f (%.3f). \n \n',...
        Str.theta(1),Str.StdErr(1),Str.theta(2),Str.StdErr(2));
fprintf('\n The lamda from the Poisson structural estimates are (taken as the mean): lamda_struct = %.3f. \n \n', Str.lambda);
fprintf('\n The ML Estimate for the Poisson reduced form model is: lamda_rf = %.3f (%.3f). \n \n', RF.lambda1,RF.StdErr2);
fprintf('\n The mean of our dependant variable is: mean(Y) = %.3f. \n \n', mean(variable.Y));




