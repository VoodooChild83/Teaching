
clear all; clc; parpool

%Load the data and clean
fname = ['/Users/idiosyncrasy58/Dropbox/Documents/College'...
         '/Universitat Autonoma de Barcelona/IDEA - '...
         'Economics/Teaching Assistant/GSE Tutoring - '...
         'Econometrics/Practice/Creel Homework Assignments '...
         'GSE/PS2/NerloveData.m'];
         
data = load(fname);

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

%The anonymous function:
%here theta is the parameter vector that includes 
%both the betas and the sigma (variance of the model).
%Use end-1 to select the set of parameters
%that correspond to the betas, since the last object
%is the sigma parameter

%Remember: WE ARE MAXIMIZING -> Obective Function must be 
%multiplied by -1 so that FMINUNC maximizes (by the 
%property that multiplying a function by -1 generates
%the reflection)

L = @(theta) -1*(-(N/2)*log(2*pi*theta(end).^2) - (1/(2*theta(end).^2))*...
            (Y-X*theta(1:end-1))'*(Y-X*theta(1:end-1)));

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

%The starting value is a random vector
%recall that we need to create a (K+1)x1
%vector since the '1' above is the sigma
%parameter of the errors.

beta_0 = ones(K+1,1);

[beta_hat] = fminunc(L,beta_0,options)

%We are asked for sample sizes N=5,50,500. Let's create a
%vector of these samples sizes (I did this with an extra
%sample size to get nicer graph outputs).

N=[5;50;500;5000];

%population probability
p_true = 0.2;

%Simulation run
T=1000;

%Initialize the vectors for the output of the Monte Carlo
%We will initialize a vector of 1000x4, since we have 4
%sample sizes, that is the size of the N vector we created
%is 4x1.
p_hat = NaN(T,size(N,1));

%A starting value of guess for the p_hat that we will find
p_0 = rand;

%Our options for the maximization routine:

options_2 = optimoptions('fmincon','Algorithm','interior-point',...
                         'Display','off','UseParallel',true);

%Program the Monte-Carlo experiment

%outer loop runs the sample sizes
tic
for i=1:size(N,1)

    sample_size = N(i);
    
    %run inner loop = MC loop - to make this go fast 
    %we will need to parallelize this otherwise it will
    %take forever
    
    parfor j=1:T
    
        %First we generate a random sample of points
        %between 0 and 1. We use the rand function 
        %to do this
        
        Y_raw = rand(sample_size,1);
        
        %Now we need to create the Y vector such that it
        %takes the value 1 if the probability is less than
        %or equal to 0.2, and 0 otherwise.
        
        Y = (Y_raw <= p_true);
        
        %Now we can run the fminunc routine on the sample Y 
        %and find the p that maximizes the likelihood of 
        %observing our sample vector Y:
        
        p_hat(j,i) = fmincon(@(p) Bernoulli(p,sample_size,Y),...
                             p_0,[],[],[],[],0,1,[],options_2)
    
    end

end
toc

p_mean = mean(p_hat)
p_SE = std(p_hat)

%To implement the solution we will vectorize:
%Since N is a 3x1 vector I will create a 3x3
%matrix with all the elements on the diagonal.
%I can take the square root of this and multiply
%this to the transpose of p_hat-ptrue (which becomes
%3x1000 when transposed). This gives a 3x1000
%matrix where the square root of each sample size
%was multiplied as a scalar to the row of the 
%associated estimates that used that sample size.

p_centered = (sqrt(diag(N))*(p_hat - p_true)')';

var_p_true = p_true - p_true^2

E_p_centered = mean(p_centered)
var_p_centered = var(p_centered)

%Generate the histogram of the monte-carlo exercise with a 
%normal distribution super-imposed to see how the sample
%sizes are in relation to the normal. We will create
%subplots.

%First, let's set the bins (how the data is grouped)
bins=30;

%Second, let's find the global minimum and the maximum across 
%all the sameple sizes to set the same axis across all 
%the subplots beta_sim_1 to more easily compare how the dispersion
%(shape of the bell curve) of the estimates change as N changes. 
%Since we have two dimensions we should use nested min and max 
%statements to get the global value.

x_axis_min = min(min(p_centered));
x_axis_max = max(max(p_centered));

%Finally, we run through a loop across the columns of p_centered
%to generate a plot of the distribution of each column (so there 
%will be 4 subplots).

for i=1:size(N,1)

    subplot(2,2,i);
    histfit(p_centered(:,i),bins)
    title(['T=1000 & N=' num2str(N(i))], 'FontSize',8)
    xlabel('MLE estimates of p_0','FontSize',8)
    ylabel('Density', 'FontSize',8)
    xlim([x_axis_min x_axis_max])
    
end

delete(gcp('nocreate'))

usematlab = true;

% weekly close price of NSYE, data provided with GRETL
load nysewk.mat;
n = size(nysewk);
y = 100 * log( nysewk(2:n) ./ nysewk(1:n-1) );
data = y;
plot(y);

%%%%%%%%%%%%%%% Unconstrained maximization of logL  %%%%%%%%%%%
% note that the objective has a minus sign in front, as fminunc
% minimizes, but we want to maximize the logL
thetastart = [mean(y); var(y); 0.1];
[thetahat, logL] = fminunc(@(theta) -arch1(theta, data), thetastart);

logL = -logL; % re-convert

%%%%%%%%%%%%%%%%   Results   %%%%%%%%%%%%%%%%
fprintf('ARCH(1) results for NYSEWK data set (fminunc)\n');

fprintf('the estimate\n');
disp(thetahat);

fprintf('the logL value');
disp(logL);

BIC = -2*logL + size(thetahat,1)*log(size(y,1));
AIC = -2*logL + size(thetahat,1);
fprintf('BIC value');
disp(BIC);
fprintf('AIC value');
disp(AIC);

%%%%%%%%%%%%%%% Constrained maximization of logL  %%%%%%%%%%%
%  ARCH model needs parameter restrictions for stationariry and positive variance
%  fmincon can be used to impose them
lb = [-Inf; 0; 0];
ub = [Inf; Inf; 1];
if usematlab
  [thetahat, logL] = fmincon(@(theta) -arch1(theta, data), thetastart, [],[], [],[], lb, ub);
else
  [thetahat, logL] = sqp(thetahat, @(theta) -arch1(theta,data),[], [], lb, ub); %Octave replacement for fmincon
end

logL_UR = -logL; % re-convert

%%%%%%%%%%%%%%%%   Results   %%%%%%%%%%%%%%%%
fprintf('ARCH(1) results for NYSEWK data set (fmincon - impose stationarity) \n');

fprintf('the estimate\n');
disp(thetahat);

fprintf('the logL value');
disp(logL_UR);

%%%%%%%% here's an example of a binding constraint: use the results to do LR test   
R = [0 1 1]; % a silly constraint: make last 2 coefficients add to 2.5
r = 2.5;
lb = [-Inf; 0; 0];
ub = [Inf; Inf; 1];
if usematlab
  [thetahat_r, logL] = fmincon(@(theta) -arch1(theta, data), thetastart, [],[],R,r, lb, ub);
else
  [thetahat_r, logL] = sqp(thetahat, @(theta) -arch1(theta,data),@(theta) R*theta-r, [], lb, ub); %Octave replacement for fmincon
end
logL_R = -logL; % re-convert

%%%%%%%%%%%%%%%%   Results   %%%%%%%%%%%%%%%%
fprintf('ARCH(1) results for NYSEWK data set (fmincon - impose silly restriction) \n');

fprintf('the estimate\n');
disp(thetahat_r);

fprintf('the logL value');
disp(logL_R);

fprintf('the LR test statistic');
disp(2*(logL_UR-logL_R));


%%
%%%%%%%%%%%%%%% Unconstrained maximization of logL  %%%%%%%%%%%
% note that the objective has a minus sign in front, as fminunc
% minimizes, but we want to maximize the logL
thetastart = [mean(y); var(y); 0.1; 0.1];
[thetahat, logL] = fminunc(@(theta) -garch11(theta, data), thetastart);

logL = -logL; % re-convert

%%%%%%%%%%%%%%%%   Results   %%%%%%%%%%%%%%%%
fprintf('GARCH(1,1) results for NYSEWK data set (fminunc)\n');

fprintf('the estimate\n');
disp(thetahat);

fprintf('the logL value');
disp(logL);

BIC = -2*logL + size(thetahat,1)*log(size(y,1));
AIC = -2*logL + size(thetahat,1);
fprintf('BIC value');
disp(BIC);
fprintf('AIC value');
disp(AIC);


usematlab = true;
% Arch1Example results for start values
thetastart = [0.16; 3.17; 0.25; 0;0;0];

%%%%%%%%%%%%%%% Constrained maximization of logL  %%%%%%%%%%%
%  ARCH model needs parameter restrictions for stationariry and positive variance
%  fmincon can be used to impose them
lb = [-Inf; 0; 0; 0; 0; 0];
ub = [Inf; Inf; 1;1;1;1];
R = [0 0 1 1 1 1]; % 
r = 1;
if usematlab
  [thetahat, logL] = fmincon(@(theta) -arch4(theta, data), thetastart, R,r, [],[], lb, ub);
else
  [thetahat, logL] = sqp(thetastart, @(theta) -arch4(theta,data),[], @(theta) r-R*theta, lb, ub); %Octave replacement for fmincon
end
logL = -logL; % re-convert

%%%%%%%%%%%%%%%%   Results   %%%%%%%%%%%%%%%%
fprintf('ARCH(4) results for NYSEWK data set (fmincon - impose stationarity) \n');

fprintf('the estimate\n');
disp(thetahat);

fprintf('the logL value');
disp(logL);


BIC = -2*logL + size(thetahat,1)*log(size(y,1));
AIC = -2*logL + size(thetahat,1);
fprintf('BIC value');
disp(BIC);
fprintf('AIC value');
disp(AIC);

