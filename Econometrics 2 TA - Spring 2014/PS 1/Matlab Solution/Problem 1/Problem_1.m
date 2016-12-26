% Alberto Ramirez
% Poisson Problem, Problem 1 in PS1
%
%
clc
clear all
%
%% Define the variables and the data set:
var.n=10;

var.x=[-1 -1 1 0 -1 -1 1 1 2 2]';      % nx1 vector
var.X=[ones(var.n,1) var.x];           % Design matrix

var.Y=[0 0 0 1 1 1 2 2 2 3]';          % for Poisson

var.Reps=1000;

%% OLS Guess

% Estimate first using OLS to create estimate of parameters. Here I will
% allow for a randomization of the OLS beta so that initial guesses are not
% always the same:

ols.theta=(var.X'*var.X)\var.X'*var.Y;

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

[Str.theta,~,~,~,~,Str.Hess]=fminunc(@(l) MLE_Poisson(l,var.X,var.Y),ols.theta,Str.options);

Str.StdErr=diag(sqrt(inv(Str.Hess)));

% Now calculate the parameter Lambda based on the structural estimates and 
% take the mean:

Str.lambda=mean(exp(-var.X*Str.theta));

%% Bootstrap

% Bootstrap the standard errors and the point estimates since we are doing
% MLE with small samples, and MLE is an asymptotic theory.

Str.thetaboot=zeros(2,var.Reps);

for i=1:2*var.Reps
    
    var.index=randi(var.n,var.n,1);
    
    var.Xboot=var.X(var.index,:);
    var.Yboot=var.Y(var.index);
    
    Str.thetaboot(:,i)=fminunc(@(l) MLE_Poisson(l,var.Xboot,var.Yboot),ols.theta,Str.options);
    
    Str.lambdaboot(:,i)=exp(-var.Xboot*Str.thetaboot(:,i));
    
end

Str.thetabootmean=mean(Str.thetaboot,2);
Str.thetabootstderr=std(Str.thetaboot,0,2);

Str.lambdabootmean=mean(mean(Str.lambdaboot));
Str.lambdabootstderr=std(mean(Str.lambdaboot,1),0,2);

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

[RF.lambda1,~,~,~,~,~,RF.Hess]=fmincon(@(l) MLE_Poisson(l,var.X,var.Y),RF.lambda1guess,[],[],[],[],0,Inf,[],RF.options);

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

[RF.lambda2,~,~,~,~,RF.Hess2]=fminunc(@(l) MLE_Poisson(l,var.X,var.Y),RF.lambda2guess,Str.options);

RF.StdErr2=sqrt(inv(RF.Hess2));

%% Bootstrap Option 1:

RF.lambdaboot=zeros(var.Reps,1);

for i=1:3*var.Reps
    
    var.index=randi(var.n,var.n,1);
    
    var.Xboot=var.X(var.index,:);
    var.Yboot=var.Y(var.index);
    
    RF.lambdaboot(i)=fmincon(@(l) MLE_Poisson(l,var.Xboot,var.Yboot),RF.lambda1guess,[],[],[],[],0,Inf,[],RF.options);
    
end

RF.lambdabootmean=mean(RF.lambdaboot);
RF.lambdabootstderr=std(RF.lambdaboot);

%% Reduced Form using the Structural Form as a Constraint: Exercise in Anonymous Functions
% 
% % We can further play with this in how we estimate the model by estimating
% % the reduced form using the structural form as a constraint. Here we would
% % need to pass in three parameter values as a guess.
% 
% % This estimation will only identify the reduced form paramter.
% 
% % Define the objective function (MLE) as an anonymous function:
% 
% log_like = @(nu) - sum(var.Y.*log(nu(1)) - nu(1));
% 
% % Write the structural equation as an anonymous function. Here we require a
% % scalar, but the function will give us an Nx1 vector. Since the lambda is
% % a parameter such that in expectation all values of Y attain the value of
% % lambda, we take the mean of the structural equation such that it is equal
% % to 0.
% 
% structural = @(nu) sum(exp(var.X*nu(2:3))-nu(1));
% 
% % Use the 'deal' function to deal out the results of the estimation
% 
% st_constraint = @(nu) deal([],structural(nu));
% 
% % generate a guess such that parameter one in the vector (lambda) is
% % positive; the remaining two parameters are anything
% 
% guess=[rand(1,1);randn(2,1)];
% 
% P.lambda3_rf1 = fmincon(log_like,guess,[],[],[],[],[],[],st_constraint);
% 
% P.lambda3_rf2 = fmincon(log_like,guess,[],[],[],[],[],[],@(nu) Constraint(nu,var.X,var.Y));

%% Display results

fprintf('\n The ML Estimators for the Poisson structural model are: theta1 = %.3f (%.3f) and theta2 = %.3f (%.3f). \n \n',...
        Str.theta(1),Str.StdErr(1),Str.theta(2),Str.StdErr(2));
fprintf('\n The lamda from the Poisson structural estimates are (taken as the mean): lamda_struct = %.3f. \n \n', Str.lambda);
fprintf('\n The ML Estimate for the Poisson reduced form model is: lamda_rf = %.3f (%.3f). \n \n', RF.lambda1,RF.StdErr2);
fprintf('\n The mean of our dependant variable is: mean(Y) = %.3f. \n \n', mean(var.Y));




