
%clear all variables
clear all;

%reset the RNG to default
rng('default');

%start parallel processing
parpool

%set the constant, a:
a=0;

%set the anonymous function to maximize
f = @(x) sin(x)+0.01*(x-a).^2;

fplot(f,[-5*pi,5*pi]);

%set the options
%1. The algorithm will be quasi-newton (this is because the 
%   default algorithm, 'trust-region', requires the user to
%   input the gradient of the function as a second function),
%2. We will turn the display off (so we don't have a bunch of
%   crap on the screen).
%3. We will ask that the hessian use the 'bfsg' algorithm to 
%   update the search pattern.

options = optimoptions(@fminunc,'Algorithm','quasi-newton',...
                       'Display','off','HessUpdate','bfgs');

%set the starting value for guessing (a random integer between
%-20 and 20. 
x0 = 2.5;

%write the optimization routine
[min_val,f_val] = fminunc(f,x0,options)



%program the gradient and hessian as anon functions
%we use the deal method to have it deal out the 
%different functions

f_grad_hess = @(x) deal(sin(x)+0.01*(x-a).^2,...
                        -0.02*a + 0.02*x + cos(x),...
                        -sin(x) + 0.02);

%create a new options to tell matlab we will supply gradient
options_2 = optimoptions('fminunc','Algorithm','trust-region',...
                         'GradObj','on','Hessian','on',...
                         'Display','off');

%write the new minimization routine

[min_val_2,f_val_2] = fminunc(f_grad_hess,x0,options_2)

%number of solvers:
n_solvers=50;

%Since we have already defined the problems previously, lets recycle
%(we could use the original f function that had matlab compute approx.
%hessians and numeric gradients, but since we have easy analytical 
%solutions we can just use the f_grad_hess all the same).

%with the original problem in the first cells above, define problem
problem = createOptimProblem('fminunc','objective',f_grad_hess,...
                             'x0',x0,'options', options_2);
                             
ms = MultiStart('UseParallel', true);

%write the MultiStart routine on 50 points

[min_val_3,f_val_3] = run(ms,problem,n_solvers)

%Let's set the options for simulated annealing, we want display off
options_3 = saoptimset('Display','off');

%Write the maximization routine

%we will reuse the initial starting point from the Multistart
%problem, no need to create a new one.

[min_val_4,f_val_4] = simulannealbnd(f,x0,[],[],options_3)

%set the options for fmincon, where we provide
%user-supplied gradients and hessians:

options_4 = optimoptions('fmincon','Algorithm','trust-region-reflective',...
                         'GradObj','on','Hessian','on','Display','off',...
                         'UseParallel',true);

%the upper and lower bounds:
xlb = 0;
xub = 3;

%write the constrained minimization routine:

[min_val_5,f_val_5] = fmincon(f_grad_hess,x0,[],[],...
                              [],[],xlb,xub,[],options_4)

%Since we have already defined the problems previously, lets recycle
%We will use the associated optimization options and the objective
%function (with the programmed hessian and gradient) from f_grad_hess
%(we could also use f, but would need to adjust options to turn off
%the user-supplied gradients and hessian).

%with the original problem in the first cells above, define problem
problem_2 = createOptimProblem('fmincon','objective',f_grad_hess,...
                               'x0',x0,'lb',xlb,'ub',xub,...
                               'options', options_4);
                             
ms_2 = MultiStart('UseParallel', true);

%write the MultiStart routine on 100 points

[min_val_6,f_val_6] = run(ms_2,problem_2,n_solvers)

%We can recycle code from problem 1 and the bounds already 
%defined. We will not need to create a new set of options either.
%Once again, we will use the function f we defined previously and
%not the function with the gradient and the hessian. We will also 

[min_val_7,f_val_7] = simulannealbnd(f,x0,xlb,xub,options_3)

%Load the data
data = load('NerloveData.m');

%Remove the first column ('names' of firms):
data(:,1)=[];

%Sample Size (number of rows = number of samples)
N = size(data,1);

%Create the Y and X arrays (remember to take the logs):

%Y is column 1
Y = log(data(:,1));

%X matrix will be from data(:,2:end) and a constant
constant = ones(N,1);

X = [constant log(data(:,2:end))];

%Number of regressors:
K = size(X,2);

%Create the anonymous function that will be minimized:

SSR = @(beta) (Y - X*beta)'*(Y - X*beta);

%Create the initial value of the betas (we have 5). We
%will just start with a vecotr of 0s.

beta_0 = zeros(K,1);

%Create the minimization routine:
%we can use the same options we used for our function f 
%in problem 1 since we use the same routine.

[beta_hat,SSR_min] = fminunc(SSR,beta_0,options)

%Generate the epsilons and the SE of the betas
eps = Y-X*beta_hat;

%model variance:
var = eps'*eps/(N-K);

%the standard deviation of the model:
sigma_hat = sqrt(var)

%Variance-Covariance matrix of the beta parameters:
Var_Cov_beta = var*inv(X'*X);

%The standard errors of the betas:
SE_beta = sqrt(diag(Var_Cov_beta))

%shut down parallel computing
delete(gcp('nocreate'))
