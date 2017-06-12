function [ IRF_plot, F ] = BS_IRF( data,p,constant,reps,level,model,horizon,H )
%% DOCSTRTING
%
%This function will generate the bootstrap confidence intervals for the
%impulse response function. The output is a tensor with all the shocks'
%IRF and the upper and lower confidence bands based on the indicated
%confidence level
%
%INPUT: 
%
%Data: a Tx(K+1) data (T = the time dimension, including presample; K = the
%number of dependant variables that will be lagged (+1 because the data must
%include a column of timestamps))
%
%horizon: the horizon length for the Wold decomposition
%
%p: the number of lags, presumably determined beforehand by some
%statistical criterion test
%
%constant: 1 (any other number) if the model is to include a constant (or
%not a constant)
%
%reps: the number of repetitions the bootstrap is to do
%
%model: the same as that found in IRF.m, the type of structural modeling
%restrictions to engage (reproduced here)
%
%   'NonStruct' = nonstructural VAR
%   'Chol Decomp' = structural VAR with the Cholesky short run 0
%                       restriction
%   'Cholesky Longrun' = impose the Cholesky longrun restriction
%   'Blanchard' = impose the blanchard restriction
%
%level: the confidence level (must be a number between (0,1))
%
%H: KxK orthonormal array of restrictions
%
%OUTPUT:
%
%IRF_plot: a tensor (horizon x 3 x 4) with the plottable IRFs. The second
%dimension here (3) refers to the three IRFs that are included for each
%shock (in order: Lower Bound, IRF, Upper Bound)
%
%F: original tensor from the IRF function
%
%FUNCTIONS USED:
%VAR.m
%Companion.m
%IRF.m

%% Prep the Data

%remove the data column
[T,K] = size(data(:,2:end));

T = T-p;                          % remove the lags from the observations

F_boot = zeros(horizon,K*K,reps);

quant_UB = zeros(horizon,K*K);
quant_LB = zeros(horizon,K*K);

IRF_plot = zeros(horizon,3,K*K);

%% Estimate the VAR(p) to get the initial OLS estimates

[C,PI,~,ehat] = VAR(data,p,constant);

%% Generate the bootstrap of beta:

% Start from the first p obs of Y (and reshape into 1x(K*p) vector (VEC)
Y1 = reshape(flipud(data(1:p,2:end))',1,[]);                   

for i=1:reps
    
    % Generate the new sample
    
    Y2=zeros(T,K);
    
    index=randi(T,T,1);                             % Generate the draws
    
    for t=1:T
    
        Y2(t,:)= C' + Y1*PI' + ehat(index(t),:);
        
        Y1=[Y2(t,:) Y1(1:K*(p-1))];                 % Update Y1 so that the new value is included
        
    end
    
    % Construct the new data matrix for estimating the beta
    % Add back the necessary components to data for VAR estimation 
    % (the date column and the first p observations).
    
    databoot=[data(:,1) [data(1:p,2:end);Y2]];      
    
    % Construct and save the IRFs
    if nargin>7
      % if an orhonormal array of restrictions is included then pop it in
      [~,F_boot(:,:,i)]=IRF(databoot,horizon,p,constant,model,H);
    else
      [~,F_boot(:,:,i)]=IRF(databoot,horizon,p,constant,model);
    end
         

end

%% Create the quantiles of the IRFS

% Rearrange the tensor into a (reps x IRF x horizon) tensor to grab the quantiles

F_boot = permute(F_boot,[3 2 1]);        

for j=1:horizon

   quant_UB(j,:)=quantile(F_boot(:,:,j),level);
   quant_LB(j,:)=quantile(F_boot(:,:,j),1-level);

end  

%% Generate the original IRF from the data

if nargin>7
  %if an orthonormal array of restrictions is included pop it in
  [~,Fplot,F]=IRF(data,horizon,p,constant,model,H);
else
  [~,Fplot,F]=IRF(data,horizon,p,constant,model);
end

%% Create the plottable IRFs with the confidence bands

for i = 1:K*K

    IRF_plot(:,:,i) = [cumsum(quant_LB(:,i)) cumsum(Fplot(:,i)) cumsum(quant_UB(:,i)) ];
    
end

end