function [ L,Grad ] = MLE_Poisson( beta,X,Y )
% This function will generate the log likelihood of the Poisson model, with
% a lambda parameterized by thetas and an X variable; or a lamda provided
% directly (if reduced form is desired).

% Run an if statement to capture the possibility that we may want to
% estimate a reduced form of the model.

N=size(Y,1);

if numel(beta)>1                     

   % Define the Lambda Parameter

    lambda = exp(-X*beta);
    
else
    
    lambda = beta;
    
end

% Define the log-likelihood function

log_like = Y.*log(lambda) - lambda;

% Take the sum and make negative for maximization

L = -1/N*sum(log_like);

% For the case in a reduced form regression, generate the gradient
% analytically - that is, the score. This will be used by fmincon and a
% specific algorithm that requires the use of a user-supplied gradient.

if numel(beta)==1
    
   Grad=-1/N*sum(Y/lambda - 1);
   
else
   
   Grad=-1/N*(X'*(Y-lambda));
   
end


end

