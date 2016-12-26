% Alberto Ramirez 
% Problem 2 and 3 in PS1
% Endogenous regression of autocorrelated variables with an exogenous
% variable.

clc
clear all
tic
rng('default');
%% Declaration of Variables
% Here we wish to estimate an AR2 process of normally distributed errors
% and an IIN exogenous variable.

% Simulation of data: Construct DGP array to hold the true values of the
% coefficients and var array to hold the values of the dataset.

var.N=input('Please enter a sample size to simulate: N = '); % Number of Observations
          if     var.N <= 0;
                 var.N = 30;
          elseif isempty(var.N);
                 var.N = 30;
          elseif   var.N > 0;
                 var.N;
          end
          
var.lags=2;

var.q=round(0.75*(var.N-var.lags)^(1/3));
          
var.X=randn(var.N,1);                   % Vector of exogenous IIN data points
var.e=randn(var.N,1);                   % Vector of normally distributed errors
var.constant=ones(var.N-var.lags,1);    % Vector for the constant term

var.Reps=1000;                           % for the Monte-Carlo Simulation

% To generate a simulation of the dependant variable from the true model
% parameters, I will use the array DGP to collect all the DGP variables.

DGP.alpha=0;
DGP.rho1=0.5;
DGP.rho2=0.4;
DGP.beta=2;
Param=[DGP.alpha;DGP.rho1;DGP.rho2;DGP.beta];

%% Moment System

% Define the moment system:

Moment.e=@(m,Y,X) Y-X*m;

Moment.Moment = @(m,Y,X,Z) Z'*Moment.e(m,Y,X);

Moment.MC = @(m,Y,X,Z) diag(Moment.e(m,Y,X))*Z;

%% Generation of Dependant Variable
% We will use the following AR2 DGP to produce the vector of dependent
% variables: y = alpha + rho1 * y(t-1) + rho2 * y(t-2) + beta * x + e
% We recognize that Y will be normally distributed since the error is normally distributed. 
% To create the Y vector through the DGP process we must initialize the two
% observations to generate the remaining N oservations of Y. We may choose
% to use either to draws from the random normal, or two 0

DGP.y=zeros(var.N,1);%[randn(2,1);zeros(var.N,1)];

for j=3:var.N;
    DGP.y(j,1)=DGP.alpha+DGP.rho1*DGP.y(j-1)+DGP.rho2*DGP.y(j-2)...
              +DGP.beta*var.X(j)+var.e(j);
end

var.Yt=lagmatrix(DGP.y,0:var.lags);
var.Y=var.Yt(var.lags+1:end,1);
var.Yt1=var.Yt(var.lags+1:end,2);
var.Yt2=var.Yt(var.lags+1:end,3);

% The below code implements an incorrect method of generating the Y
% vectors - that is, in this method, Y does not follow the DGP and so
% regression results will not make sense. 

% DGP.y=randn(var.N+var.lags,1);
% var.Y=DGP.y(var.lags+1:end);
% var.Yt1=lagmatrix(DGP.y,1);
% var.Yt2=lagmatrix(DGP.y,2);
% var.Yt1(1:var.lags,:)=[];
% var.Yt2(1:var.lags,:)=[];

%% IV Instruments:
% Here we will generate the IV instrument matrices.

% First lag the exogs:
IV.Xt=lagmatrix(var.X,0:var.lags);
var.X=IV.Xt(var.lags+1:end,1);
IV.Xt1=IV.Xt(var.lags+1:end,2);        % for the AR2 instrument
IV.Xt2=IV.Xt(var.lags+1:end,3);        % for the AR1 instrument

% Now the instruments for each model:
IV.AR1.Z=[var.constant var.X IV.Xt1];         % three instruments
IV.AR2.Z=[var.constant var.X IV.Xt1 IV.Xt2];  % four instruments

%% Estimation of incorrectly specified AR1 model by OLS
% Construct the data matrices for OLS Estimation. We will use the array AR1
% to collect variables for this estimation.

AR1.X=[var.constant var.Yt1 var.X];
AR1.coeff=(AR1.X'*AR1.X)\AR1.X'*var.Y;
AR1.coeff2=[AR1.coeff(1:2,1);NaN;AR1.coeff(3,1)];       % This vector will allow us to printout the results

[AR1.Avar]=HAC(AR1.coeff,var.Y,AR1.X,AR1.X,Moment.MC,Moment.e);

AR1.StdErr=sqrt(diag(AR1.Avar));
AR1.StdErr=[AR1.StdErr(1:2);NaN;AR1.StdErr(3)];         % This is for the printout of the results. 


%% Estimation of correctly specified AR2 model by OLS
% We will simply add the final lag to the previous data matrix. We will
% also use the array AR2 to collect these variables for this estimation.

AR2.X=[var.constant var.Yt1 var.Yt2 var.X];
AR2.coeff=(AR2.X'*AR2.X)\AR2.X'*var.Y;

[AR2.Avar]=HAC(AR2.coeff,var.Y,AR2.X,AR2.X,Moment.MC,Moment.e);

AR2.StdErr=sqrt(diag(AR2.Avar));

%% IV estimation for the AR1 process

IV.AR1.coeff=(IV.AR1.Z'*AR1.X)\IV.AR1.Z'*var.Y;
IV.AR1.coeff2=[IV.AR1.coeff(1:2,1);NaN;IV.AR1.coeff(3,1)];

var.q=round(0.75*var.N^(1/3));

[IV.AR1.Avar]=HAC(IV.AR1.coeff,var.Y,AR1.X,IV.AR1.Z,Moment.MC,Moment.e,var.q);

IV.AR1.StdErr=sqrt(diag(IV.AR1.Avar));
IV.AR1.StdErr=[IV.AR1.StdErr(1:2);NaN;IV.AR1.StdErr(3)];

%% IV estimation for the AR2 process

IV.AR2.coeff=(IV.AR2.Z'*AR2.X)\IV.AR2.Z'*var.Y;

[IV.AR2.Avar]=HAC(IV.AR2.coeff,var.Y,AR2.X,IV.AR2.Z,Moment.MC,Moment.e);

IV.AR2.StdErr=sqrt(diag(IV.AR2.Avar));

%% Problem 3 GMM Estimation of the AR1 process: 2-Step GMM
      
% A function has been written that will calculate the efficient 2-step
% estimation of the GMM estimator (which is techincally still IV if we use
% the same instruments as we did in the IV estimation for the AR1). Hence for this estimation of
% AR1 by GMM I will use the full Instruments from AR2:

EGMM.Z=[IV.AR1.Z IV.Xt1.^2];% var.X.^2 var.X.^3 IV.Xt1.^3];

% Estimate GMM

[EGMM.delta,EGMM.StdErr]=GMM(AR1.coeff,var.Y,AR1.X,EGMM.Z,Moment.Moment,Moment.MC,Moment.e,var.q);

EGMM.delta=[EGMM.delta(1:2);NaN;EGMM.delta(3)];
EGMM.StdErr=[EGMM.StdErr(1:2);NaN;EGMM.StdErr(3)];

%% Monte Carlo Simulation
% Now we simulate estimates in a loop by constructing a monte carlo
% simulation for 1000 replicates of all the estimations done previously:

[Simul.AR1.rho1,Simul.AR2.rho1,Simul.AR2.rho2,Simul.AR1IV.rho1,Simul.AR2IV.rho1,Simul.AR2IV.rho2, Simul.GMM.rho1]=...
    MC(var.N,var.Reps,Param,var.lags,Moment.Moment,Moment.MC,Moment.e);

Simul.AR1.coeff2=[NaN;mean(Simul.AR1.rho1)+DGP.rho1;NaN;NaN];
Simul.AR1.StdErr=[NaN;std(Simul.AR1.rho1);NaN;NaN];

Simul.IV.AR1.coeff2=[NaN;mean(Simul.AR1IV.rho1)+DGP.rho1;NaN;NaN];
Simul.IV.AR1.StdErr=[NaN;std(Simul.AR1IV.rho1);NaN;NaN];

Simul.AR2.coeff=[NaN;mean(Simul.AR2.rho1)+DGP.rho1;mean(Simul.AR2.rho2)+DGP.rho2;NaN];
Simul.AR2.StdErr=[NaN;std(Simul.AR2.rho1);std(Simul.AR2.rho2);NaN];

Simul.IV.AR2.coeff=[NaN;mean(Simul.AR2IV.rho1)+DGP.rho1;mean(Simul.AR2IV.rho2)+DGP.rho2;NaN];
Simul.IV.AR2.StdErr=[NaN;std(Simul.AR2IV.rho1);std(Simul.AR2IV.rho2);NaN];

Simul.GMM.coeff2=[NaN;mean(Simul.GMM.rho1)+DGP.rho1;NaN;NaN];
Simul.GMM.StdErr=[NaN;std(Simul.GMM.rho1);NaN;NaN];

toc
%% Printout of the AR1, AR2, and AR IV and GMM Point Estimations

disp.Tags(:,1)={'alpha','rho1','rho2','beta'};

% For point estimation:
disp.model=cell(5,6);
disp.model(1,2:6)={'AR1 OLS','AR1 IV','AR2 OLS','AR2 IV','AR1 GMM'};
for i=1:4;
    disp.model(i+1,1)=disp.Tags(i);
    disp.model(i+1,2:end)={AR1.coeff2(i),IV.AR1.coeff2(i),AR2.coeff(i),IV.AR2.coeff(i),EGMM.delta(i)};
end
display('Point Estimation of Model Parameters for OLS and IV estimation');
display(disp.model);

% Printout of the Point Estimate Standard Errors
disp.StdErr=cell(5,6);
disp.StdErr(1,2:6)={'AR1 OLS','AR1 IV','AR2 OLS','AR2 IV','AR1 GMM'};
for i=1:4;
    disp.StdErr(i+1,1)=disp.Tags(i);
    disp.StdErr(i+1,2:end)={AR1.StdErr(i),IV.AR1.StdErr(i),AR2.StdErr(i),IV.AR2.StdErr(i),EGMM.StdErr(i)};
end
display('Estimation of Point Estimate Standard Errors for OLS and IV estimation');
display(disp.StdErr);

%% MC Simulation Estimate Results

% Printout for MC Simulation estimates:
disp.model2=cell(5,6);
disp.model2(1,2:end)={'AR1 OLS','AR1 IV','AR2 OLS','AR2 IV','AR1 GMM'};
for i=1:4;
    disp.model2(i+1,1)=disp.Tags(i);
    disp.model2(i+1,2:end)={Simul.AR1.coeff2(i),Simul.IV.AR1.coeff2(i),Simul.AR2.coeff(i),Simul.IV.AR2.coeff(i),Simul.GMM.coeff2(i)};
end
display('MC Estimation of Model Parameters for OLS and IV estimation');
display(disp.model2);

% Printout of the Point Estimate Standard Errors
disp.StdErr2=cell(5,6);
disp.StdErr2(1,2:end)={'AR1 OLS','AR1 IV','AR2 OLS','AR2 IV','AR1 GMM'};
for i=1:4;
    disp.StdErr2(i+1,1)=disp.Tags(i);
    disp.StdErr2(i+1,2:end)={Simul.AR1.StdErr(i),Simul.IV.AR1.StdErr(i),Simul.AR2.StdErr(i),Simul.IV.AR2.StdErr(i),Simul.GMM.StdErr(i)};
end
display('Estimation of MC Standard Errors for OLS and IV estimation');
display(disp.StdErr2);

%% Printout of Histrograms for the Endogenous Parameters Simulated through the Monte Cristo loop.

disp.figure=figure('Units','normalize','Position',[0 0 1 1]); orient landscape;
subplot(3,2,1); hist(Simul.AR1.rho1,25)
title('AR1   OLS   Rho1   Sampling   Error   Distribution');
subplot(3,2,2); hist(Simul.AR1IV.rho1,25)
title('AR1   IV    Rho1   Sampling   Error   Distribution');
subplot(3,2,3); hist(Simul.AR2.rho1,25)
title('AR2   OLS   Rho1   Sampling   Error   Distribution');
subplot(3,2,4); hist(Simul.AR2.rho2,25)
title('AR2   OLS   Rho2   Sampling   Error   Distribution');
subplot(3,2,5); hist(Simul.AR2IV.rho1,25)
title('AR2   IV    Rho1   Sampling   Error   Distribution');
subplot(3,2,6); hist(Simul.AR2IV.rho2,25)
title('AR2   IV    Rho2   Sampling   Error   Distribution');

