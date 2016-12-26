% Alberto Ramirez
% Problem 4

% Estimation of Investment equation for Klein Model using GMM
%
% We will estimate the following model:
%
% Investment I = beta + beta1*P(t) + beta2*P(t-1) + beta3*K(t-1) + e(t)
% and use the following list of instrumental variables
% With lagged endogs (assumption is nonautocorrelated errors): [1,W(g),G,T,A,P(t-1),K(t-1),X(t-1)]
% With no endogs (assumption is there is autocorrelation

clc
clear all

%% Data Set: Variable Extraction

% In this section load the data from the text file and define them within
% a variable array.

var.data=load('klein.data.txt');
var.N=length(var.data);

var.q=round(0.75*(var.N)^(1/3));        % Stock and Watson rule of thumb for calculating bandwidth

% Extract the relevant variables from the data matrix and assign variable
% names:

% Dependant variable
var.I=var.data(:,5);

% Regressor matrix, adding also a lag to the P(t) regressor

var.X=[ones(var.N,1) var.data(:,3)... 
            lagmatrix(var.data(:,3),1) var.data(:,6)];

% Now we grab the Instruments and place them directly into the instrument
% data matrix. 

Inst.Z1=[var.X(:,[1,3]) var.data(:,[6,8,9,10])...
              var.data(:,1)-1919 ...
              lagmatrix(var.data(:,7),1)];
          
Inst.Z2=[var.X(:,1) var.data(:,[8,9,10]) var.data(:,1)-1919];

% Since we are lagging the P regressor, we must drop the first observation
% for all our data matrices and vectors.

var.X=var.X(2:end,:);
Inst.Z1=Inst.Z1(2:end,:);
Inst.Z2=Inst.Z2(2:end,:);
var.I=var.I(2:end);
          
%% OLS for Guess:

guess=(var.X'*var.X)\var.X'*var.I;
%guess=ones(4,1);

%% 2-Step GMM Estimation of Model

% I collect all GMM modeling variables into the EGMM structured array.

% I will use my Two-Step GMM Function to calculate the estimates of the
% model. I will also use a bandwidth of 2 for the Newey-West estimator:

% Define the moment system

Moment.e=@(m,Y,X) Y-X*m;

Moment.Moment = @(m,Y,X,Z) Z'*Moment.e(m,Y,X);

Moment.MC = @(m,Y,X,Z) diag(Moment.e(m,Y,X))*Z;

% Estimate GMM:

[EGMM1.delta, EGMM1.StdErr,EGMM1.Avar, EGMM1.J,~,EGMM1.h2,EGMM1.h1]=GMM(guess,var.I,var.X,Inst.Z1,Moment.Moment,Moment.MC,Moment.e);

[EGMM2.delta, EGMM2.StdErr,EGMM2.Avar, EGMM2.J,~,EGMM2.h2,EGMM2.h1]=GMM(guess,var.I,var.X,Inst.Z2,Moment.Moment,Moment.MC,Moment.e,var.q);

%% Statistics Section 1

% t statistics
for i=1:size(var.X,2)
    stats1.t(i)=EGMM1.delta(i)/EGMM1.StdErr(i);
end

stats1.criticalt=icdf('t',0.975,size(var.data,1)-1-size(var.X,2));

for i=1:size(var.X,2)
    stats1.UB(i)=EGMM1.delta(i)+stats1.criticalt*EGMM1.StdErr(i);
    stats1.LB(i)=EGMM1.delta(i)-stats1.criticalt*EGMM1.StdErr(i);
end

% Calculate the Wald Statistic. Here we will assume that, as in the above t
% statistic (which was implicit) the null hypothesis is that each of our
% regressors is equal to 0.

% Generate the R and r matrix: R=4x4 matrix with ones on the main diagnol
% and r=4x1 vector of zeros.

stats1.R=eye(size(var.X,2));
stats1.r=zeros(size(stats1.R,1),1);

stats1.criticalchi=icdf('chi2',0.95,size(stats1.r,1));

stats1.W=(stats1.R*EGMM1.delta-stats1.r)'*((stats1.R*EGMM1.Avar*stats1.R')\eye(size(stats1.R,1)))*(stats1.R*EGMM1.delta-stats1.r);

%% Statistics Section 2

% t statistics
for i=1:size(var.X,2)
    stats2.t(i)=EGMM2.delta(i)/EGMM2.StdErr(i);
end

for i=1:size(var.X,2)
    stats2.UB(i)=EGMM2.delta(i)+stats1.criticalt*EGMM2.StdErr(i);
    stats2.LB(i)=EGMM2.delta(i)-stats1.criticalt*EGMM2.StdErr(i);
end

% Calculate the Wald Statistic. Here we will assume that, as in the above t
% statistic (which was implicit) the null hypothesis is that each of our
% regressors is equal to 0.

% Generate the R and r matrix: R=4x4 matrix with ones on the main diagnol
% and r=4x1 vector of zeros.

stats2.R=eye(size(var.X,2));
stats2.r=zeros(size(stats2.R,1),1);

stats2.criticalchi=icdf('chi2',0.95,size(stats2.r,1));

stats2.W=(stats2.R*EGMM2.delta-stats2.r)'*((stats2.R*EGMM2.Avar*stats2.R')\eye(size(stats2.R,1)))*(stats2.R*EGMM2.delta-stats2.r);

%% Outpout the Klein Model estimates

disp.Tags(:,1)={'constant','P','P(t-1)','K(t-1)'};

disp.model1=cell(5,6);
disp.model1(1,1:end)={'','coefficients','Std Err','t-value','CI-LB','CI-UB'};
    for i=1:4;
        disp.model1(i+1,1)=disp.Tags(i);
        disp.model1(i+1,2:6)={EGMM1.delta(i),EGMM1.StdErr(i),stats1.t(i),stats1.LB(i),stats1.UB(i)};
    end
    fprintf('Estimation of Model Parameters: Dependent Variable = I (assume no autocorrelation)\n\n');
    fprintf('Instruments = [1,W(g),G,T,A,P(t-1),K(t-1),X(t-1)] \n\n');
    display(disp.model1);
    fprintf('\n')
    fprintf('The t-students critical value based on v = %0.0f degress of freedom is %0.3f. \n\n',size(var.data,1)-1-size(var.X,2),stats1.criticalt)
    fprintf('The model is jointly significant as W is %0.2f and the critical value is %0.2f. \n\n',stats1.W,stats1.criticalchi)
    
    fprintf('***************************************************************************************************\n\n');

disp.model2=cell(5,6);
disp.model2(1,1:end)={'','coefficients','Std Err','t-value','CI-LB','CI-UB'};
    for i=1:4;
        disp.model2(i+1,1)=disp.Tags(i);
        disp.model2(i+1,2:6)={EGMM2.delta(i),EGMM2.StdErr(i),stats2.t(i),stats2.LB(i),stats2.UB(i)};
    end
    fprintf('Estimation of Model Parameters: Dependent Variable = I (assume autocorrelation)\n\n');
    fprintf('Instruments = [1,W(g),G,T,A] \n\n');
    display(disp.model2);
    fprintf('\n')
    fprintf('The t-students critical value based on v = %0.0f degress of freedom is %0.3f. \n\n',size(var.data,1)-1-size(var.X,2),stats1.criticalt)
    fprintf('The model is jointly significant as W is %0.2f and the critical value is %0.2f. \n\n',stats2.W,stats2.criticalchi)







