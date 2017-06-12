function [ mu,C_plot,C,Omega ] = IRF( data,horizon,p,constant,model,H)
%% Docstring
%
%This function generates a matrix of the impulse response function for
%horizon lengths that are entered into the function (temporary?: will remove
%this and replace with a while loop, as the Wold decomposition is
%stationary if the VAR is stable --> at some point the series converges to
%numerical tolerance and so no need to feed in a horizon, use a counter to
%find the horizon length within the while loop to then feed to other parts
%of the function.)
%
%INPUT:
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
%model: The type of model to use. 
%
%   'NonStruct' = nonstructural VAR
%   'Chol Decomp' = structural VAR with the Cholesky short run 0
%                       restriction
%   'Cholesky Longrun' = impose the Cholesky longrun restriction
%   'Blanchard' = impose the blanchard restriction
%
%H: KxK orthonormal array of restrictions
%
%OUTPUT:
%mu: Kx1 array with the unconditional means of the process
%
%C: KxKxHorizon array of the impulse responses
%
%C_plot: horizon x K flattened array of C for easier plotting
%
%Omega: KxK variance-covariance matrix of the model estimated by OLS
%
%FUNCTIONS USED:
% VAR.m
% Companion.m


%% Dimensions of the arrays

[T,K] = size(data);

K = K-1;                          % Remove the lags from the observation

%% Estimate the VAR(p) model that will be inverted

[Const,PI,~,ehat] = VAR(data,p,constant);

%% Generate the Companion Form

A = Companion(PI);

%% The Unconditional Mean of the Process

Const = [Const;zeros(length(PI)-length(Const),1)];

mu = (eye(size(A))-A)\Const;

mu = mu(1:K);                         % Grab the Kx1 elements of mu

%% Variance-Covariance of the Model

Omega = (ehat'*ehat)./(T-K*p-1);
   
%% Generate the Wald decomposition (VMA(Inf)) Representation

% Define the tensor to store the matrices C (where in 
% structural estimation this is actually the matrix F). Since the process 
% is stationary, it will generally be well approximated for a large horizon, h.

Wold = zeros(size(A,1),size(A,2),horizon);

for i=1:horizon
    
    Wold(:,:,i) = A^(i-1);       % fill in each tensor level with the power
    
end

% Impulse responses
C = Wold(1:K,1:K,:);

%% Generate the S array for the approriate model (and H if not included)

% Create the Cholesky decomposition of the Omega array, adjusting as
% necessary per the model used

if strcmp('Chol Decomp',model)
    
    S = chol(Omega,'lower');
   
elseif strcmp('Chol Longrun',model) || strcmp('Blanchard',model)
    
    %create the Blanchard and Longrun Cholesky Restriction arrays
    C_sum = sum(C,3);
    
    S = chol( C_sum*Omega*C_sum' ,'lower');
    
    if strcmp('Chol Longrun',model)
    
        S = C_sum\S;
       
    end
    
else
    
   %for the 'NonStruct' model we use the identity array
   S = eye(K);
   
end

% Create the H array of orthonormal restrictions if there is no H input to
% the identity array

if nargin<6
    
    H = eye(K);
    
end

%% Asdjust the IRF 

for i=1:horizon
    
    C(:,:,i) = C(:,:,i)*S*H;
    
end

% Reshape the tensor to a 2-Dimensional matrix for plotting: h x (K*K)

C_plot=reshape(permute(C, [3 2 1]),[],K*K);

end

