function [ MSE,Data ] = VARForecast( data, p, horizon, start,  constant )
%% DOCSTRING
%
% forecast = VARForecast_2( Data, p, horizon, constant)
%
% This function generates the forecast of a VAR up to a horizon length.
%
% INPUTS: 
% Data - Tx(K+1) dataset (the length of the data should be upto the
% period t where the model is to estimated, and forecasts will start at
% t+1). The first column in Data is taken to be some sort of ID value for
% t.
%
% p - number of lags to take for the K variables (determined prior to 
% estimation of the model)
%
% start - the starting index to start the forecasts
%
% constant - 1 (any other number) if the model is to include a constant 
%(or not)
%
% horizon - either 1xh vector, or scalar valued --> forecast horizon(s).
%
% OUTPUT:
% MSE - horizon x K array of the root mean square errors
%
% Data - Tx(K*horizon+K+1) array of the original dataset augmented with the
% forecasts
%
% FUNCTIONS USED:
% VAR.m
% Companion.m

%% Prep variables

% Size of the original data array
[T,K]=size(data(:,2:end));

% Check that horizon is a vector or a scalar:
h = size(horizon);
H = max(h);

%Assign the horizon length to H based on whether horizon was a scalar or
%not
if H==1
    
    H = horizon;    %horizon is a scalar, so we put in the scalar value 
                    %back into H
    
else
    
    H = horizon(end);  %horizon is a vector --> keep just the final horizon length
    
end

% Preallocate arrays that will hold the forecast to outputs
Data = [data NaN(T,K*H)];
MSE = NaN(H,K);

%% Start the iterative forecasting of the model using the companion form

for t = 0:length(data(start:end-2,:))
    
    % Estimate the model up to time t
    [C,PI,~,~,~,Y] = VAR(data(1:start+t,:),p,constant);
    
    % Keep only the last K*p observations
    Y = Y(start+t+1-2*p:start+t-p,:);
    
    % Get Companion forms
    [A,C,Yin] = Companion(PI,C,Y);
    
    for h = 1:H
        
        % Forecast
        Yout = C + A*Yin';
        
        % Map forecast into our output data array (Don't over fill the array)
        if (start+t)<=T-h
           
           % Map back into the data
    
            if K~=1
                Data(start+t+h,2+K*h:K*h+K+1) = Yout(1:K)';
            else
                Data(start+t+h,1+K+h:K+h+1) = Yout(1:K)';
            end
        
        end 
        
        % Update Y for next forecast
        Yin = Yout';
        
    end  
      
end

%% Generate the root mean squared errors

for k=1:K
    
    % Create a Tx(K*H) repeated array of the Y variable to conduct MSE
    Y = repmat(Data(:,1+k),1,H);
    
    % Generate the SE
    Error = (Y - Data(:,(1+K+k:K:end))).^2;
    
    % Calculate the RMSE
    MSE(:,k)=sqrt(nanmean(Error)');
    
end

end

