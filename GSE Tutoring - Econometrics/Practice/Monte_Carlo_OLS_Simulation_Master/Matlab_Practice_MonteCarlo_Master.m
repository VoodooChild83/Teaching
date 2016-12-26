
clc; clear; clear all;

rng('default')

%This cell initializes the arrays and the parameters 
%of the simulation

%Vector of sample sizes:
N = [10;100;1000;10000];

%Set iterator for simulator loop:
T = 1000;

%Initialize the vectors that will hold the data:
beta_sim_0 = NaN(T,size(N,1)); % --> Creates a 1000x4 vector.
beta_sim_1 = NaN(T,size(N,1));

%Population betas:
beta = [15.5;2.45];

%This cell programs the MonteCarlo

%We will have two loops - a nested structure. The outer loop runs 
%through the vector N of sample sizes, grabs the value, creates 
%the data from the DGP that will be programmed, and then....
%The second loop will conduct the 1000 iterations of estimating 
%the parameters and collect the resulting beta estimates into 
%the pre-allocated vectors.

tic
%outer loop runs through the vector of sample sizes
for i=1:size(N,1)

    %grab the sample size
    sample_size = N(i);
    
    %Create the constant vector
    constant = ones(sample_size,1);

    %inner loop runs through the number of iterations we want to simulate
    for j=1:T

        %create the data from the DGP - the right-hand side of the 
        %above DGP equation
        x = 3*rand(sample_size,1);

        %concatinate to create the X data matrix:
        X = [constant x];
        
        %create the vector of errors (epsilons)
        epsilon = randn(sample_size,1);

        %generate the Y vector of dependent variables:
        Y = X*beta + epsilon;

        %estimate the model by OLS
        beta_sim = (X'*X)\X'*Y;
    
        %put into the vectors holding the simulation results
        beta_sim_0(j,i) = beta_sim(1);
        beta_sim_1(j,i) = beta_sim(2);
    
    end

end
toc

%Calculate the estimate of the betas and the standard error 
%from the monte-carlo run. Place it into a matrix, recalling 
%that each column corresponds to the sample size from the 
%respective run.

beta_hat_MC = [mean(beta_sim_0);mean(beta_sim_1)]

SE_MC = [std(beta_sim_0);std(beta_sim_1)]

%Calculate the MSE of the coefficients

MSE_beta_0 = var(beta_sim_0) + (beta_hat_MC(1,:) - beta(1)).^2
MSE_beta_1 = var(beta_sim_1) + (beta_hat_MC(2,:) - beta(2)).^2

%Generate the histogram of the monte-carlo exercise

%This code will generate the historgrams by creating subplots
%to view all the distributions all in one display

%First, let's set the bins (how the data is grouped)
bins=30;

%Second, let's find the global minimum and the maximum across 
%all the sameple sizes to set the same axis across all 
%the subplots beta_sim_1 to more easily compare how the dispersion
%(shape of the bell curve) of the estimates change as N changes. 
%Since we have two dimensions we should use nested min and max 
%statements to get the global value.

x_axis_min = min(min(beta_sim_1));
x_axis_max = max(max(beta_sim_1));

%Finally, we run through a loop across the columns of beta_sim_1
%to generate a plot of the distribution of each column (so there 
%will be 4 subplots).

for i=1:size(N,1)

    subplot(2,2,i);
    histogram(beta_sim_1(:,i),bins)
    title(['T=1000 & N=' num2str(N(i))], 'FontSize',8)
    xlabel('OLS estimates of \beta_1','FontSize',8)
    ylabel('Density estimate', 'FontSize',8)
    xlim([x_axis_min x_axis_max])
    
end


%Set the parameters of the population and initialize arrays

%Set the sample size:
sample_size = 100;

%Append to beta vector the new value for x_2:
beta = [beta;6];

%Initialize array (here we will store each beta in a column):
%Recall T=1000, so dim of Vector ->1000 sims x 2 parameters 
beta_sim2 = NaN(T,2);  

constant = ones(sample_size,1);

%The Monte-Carlo Cell

%Here we have only one loop since we are only doing this with
%one sample size.

tic
for i=1:T

    %program the regressors
    %x_1:
    x_1 = 3*rand(sample_size,1);
    %error term nu and x_2:
    nu = 2*randn(sample_size,1);
    x_2 = x_1 + nu;
    
    %create the data matrix for the dgp:
    X_dgp=[constant x_1 x_2];
    
    %generate the epsilon errors for Y:
    epsilon = randn(sample_size,1);
    
    %generate the dependent variable Y:
    Y = X_dgp*beta + epsilon;
    
    %estimate the incorrect model: 
    % y = beta_0 + beta_1*x_1 + epsilon
    
    X_err = X_dgp(:,1:2);
    
    %recall that (X'X)X'Y -> dim Kx1 for betas
    %But beta_sim2 is dim TxK => each row is
    %dim 1xK. So we take the transpose of (X'X)X'Y
    %to place this into beta_sim2
    beta_sim2(i,:) = ((X_err'*X_err)\X_err'*Y)';

end
toc

beta_hat_MC2 = mean(beta_sim2)
SE_MC2 = std(beta_sim2)

%Calculate the MSE of the estimate of the parameter beta_1:

MSE_beta_1 = var(beta_sim2(:,2)) + (beta_hat_MC(2) - beta(2)).^2

% The function takes on the form 
%[result,t_value,t_stat] = Sig_Test(beta_hat,beta_null,SE,alpha,SS)

%we can pick what we want to keep from the output. For example,
%if we only care about the result (whether the test was passed) we
%can keep just the output. Otherwise, we can keep all the values.

[result_MC2,t_value_MC2,t_stat] = Sig_Test(beta_hat_MC2(2),beta(2),...
                                           SE_MC2(2),0.05,sample_size)

%Plot the histograms in the same graph

%generate a histogram object for the distribution from the 
%first part of the assignment -> recall that beta_sim_1(:,2)
%are the 1000 estimates of beta_1 where the sample size N=100
%(corresponding to column 2 of the array). Recall that bins=30
%was already set above so we don't have to redefine it.
[n,x_2_MC1] = hist(beta_sim_1(:,2),bins);

%generate the histogram object of this MC run, recalling that
%column 2 of beta_sim2 contains the 1000 estimates of beta_1.
[n2,x_2_MC2] = hist(beta_sim2(:,2),bins);

%call the bar graph on the first MC run. We will use the 'hold on' 
%command to tell matlab that we will be adding stuff to the graph, 
%in this case we will add the second histogram object as a second
%bar chart, but adding it to our plot as an object 'h' so that
%we can manipulate its color and such.
bar(x_2_MC1,n,'hist')
hold on; h=bar(x_2_MC2,n2,'hist'); hold off
set(h,'facecolor','c')

%set other plot elements:
title('T=1000 & N=100 for \beta_{1} across MC1 and MC2', 'FontSize',8)
xlabel('OLS estimates of \beta_1','FontSize',8)
ylabel('Density estimate', 'FontSize',8)
legend({'\beta_{1,MC1}','\beta_{1,MC2}'})

%Reset beta and initialize arrays (we keep other 
%variables as they are)

%Reset beta vector to original problem:
beta = beta(1:2);

%Initialize array (here we will store each beta in a column):
%Recall T=1000, so dim of Vector -> 1000 sims x 2 parameters 
beta_endog = NaN(T,2); 
beta_IV = NaN(T,2);

%Cell that runs the monte-carlo experiment

for i=1:T
    
    %generate the epsilon errors for Y and endog:
    epsilon = randn(sample_size,1);

    %create the true regressor:
    x_1 = 3*rand(sample_size,1);
    %and the endogenous regressor
    x_1_endog = x_1 + epsilon;
    
    %create the instrument:
    nu = randn(sample_size,1);
    rho = 1;
    x_3 = rho.*x_1 + nu;

    %create the data matrix for the dgp:
    X_dgp=[constant x_1];

    %generate the dependent variable Y:
    Y = X_dgp*beta + epsilon;

    %estimate the biased model: 
    % y = beta_0 + beta_1*x_1_endog + epsilon

    X_bias = [constant x_1_endog];

    %recall that (X'X)X'Y -> dim Kx1 for betas
    %But beta_misest is dim TxK => each row is
    %dim 1xK, corresponding to one run of the simulation. 
    %So we take the transpose of (X'X)X'Y
    %to place this into beta_misest
    beta_endog(i,:) = ((X_bias'*X_bias)\X_bias'*Y)';

    %2 - Stage Least Sqares regression

    %step 1: 
    %estimate the regression 
    %x_1 = gamma_0 + gamma_1 x_3 + zeta

    X_1step=[constant x_3];

    gamma = (X_1step'*X_1step)\X_1step'*x_1;

    %create the predicted values of x_1, x_1_hat:
    x_1_hat = X_1step*gamma;

    %step 2
    %use this predicted value of x_1_hat in the model

    X_2step=[constant x_1_hat];

    %estimate the betas:

    beta_IV(i,:) = ((X_2step'*X_2step)\X_2step'*Y)';

end

beta_hat_endog = mean(beta_endog)
SE_endog = std(beta_endog)
beta_hat_IV = mean(beta_IV)
SE_IV = std(beta_IV)

%calculate the t-statistic for our test:

%first test:
[result_endog,t_value_endog] = Sig_Test(beta_hat_endog(2),...
                               beta(2),SE_endog(2),0.05,sample_size)
[result_IV,t_value_IV] = Sig_Test(beta_hat_IV(2),beta(2),...
                               SE_IV(2),0.05,sample_size)


%generate a histogram object for the distribution from the 
%first part of the assignment -> recall that beta_sim_1(:,2)
%are the 1000 estimates of beta_1 where the sample size N=100
%(corresponding to column 2 of the array). Recall that bins=30
%was already set above so we don't have to redefine it.
[n,x_1_true] = hist(beta_sim_1(:,2),bins);

%generate the histogram object of this MC run, recalling that
%column 2 of beta_endog and beta_IV contains the 1000 estimates 
%of our parameter of interest, beta_1.
[n2,x_1_endog] = hist(beta_endog(:,2),bins);

[n3,x_1_IV] = hist(beta_IV(:,2),bins);

%call the bar graph on the first MC run. We will use the 'hold on' 
%command to tell matlab that we will be adding stuff to the graph, 
%in this case we will add the second and third histogram objects 
%as a second and third bar chart, but adding it to our plot as an 
%object 'h' so that we can manipulate its color and other things
%as needed.
bar(x_1_endog,n2,'hist')
hold on; h=bar(x_1_IV,n3,'hist'); hold off
set(h,'facecolor','r') 
hold on; h2=bar(x_1_true,n,'hist'); hold off
set(h2,'facecolor','c') 

%set other plot elements:
title('T=1000 & N=100 for Instrumental Estimation of \beta_1',...
      'FontSize',8)
xlabel('OLS and IV estimates of \beta_1','FontSize',8)
ylabel('Density estimate', 'FontSize',8)

%for the legend, we will need to resize it. So we call upon its
%objects to manipulate them -> we want to adjust font size
[hleg1, hobj1] = legend({'\beta_{1,endog}','\beta_{1,IV}',...
                '\beta_{1,true}'},'Location','southoutside',...
                'Orientation','horizontal');
textobj = findobj(hobj1, 'type', 'text');
set(textobj,'fontsize', 8);
