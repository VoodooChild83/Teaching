function [Decomp, T_Var] = Variance_Decomp(F,periods)
%% DOCSTRING
%
% This function generates the variance decomposition of the F array (IRFs)
%
% INPUTS:
% F: KxKxHorizon array containing the IRFs
%
% periods: vector specifying the periods of interest 
%
% Outputs:
% Decomp: KxKxHorizon array containing the variance decomposition for every
% period
%
% T_Var - KxKxsize(periods) array containing the variance decomposition for 
% the specified periods  

%% Prep the Data

% Get the size of the F array
[K,~,H] = size(F); 

% Preallocate arrays
Decomp = zeros(K,K,H);                     
T_Var = zeros(K,K,max(size(periods)));  

%% Compute Squares

F2 = F.^2;

%% Conduct the Decomposition

for i = 1:H
    
    % Sum across the time horizons
    Sum_F = sum(F2(:,:,1:i),3);

    % Generate the total variance for each of the K variables
    Sum_F_row = sum(Sum_F,2);

    % Generate the decomposition
    for j = 1:K
        
        Decomp(j,:,i)=Sum_F(j,:)./Sum_F_row(j);
        
    end
    
end

%% For the desired time periods

% Remember matlab non zero indexing --> first row index (1) = impact,
% period 1 = index(2)

T_Var(:,:,:) = Decomp(:,:,periods+1);

end