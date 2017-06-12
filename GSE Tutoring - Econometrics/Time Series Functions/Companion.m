function [ Comp_Pi, Comp_C, Comp_Y ] = Companion( PI, C, Y_t )
%% Docstring
%
% This function will generate the companion form of the VAR(p) matrix based
% on the estimated matrices from the VAR function
%
% INPUT:
% PI - Kx(p*K) array of the estimated coefficients of a VAR(p) or AR(p)
%
% C - Kx1 array of the estimated coefficients (optional)
%
% Y - (K*p)x1 array of the Y_t values
%
% OUTPUT:
% Comp_Pi - (p*K)x(p*K) array of the estimated coefficients in the
% companion form
%
% Comp_C - (p*K)xK array of the estimated constants in the compaion form
%
% Comp_y - (2*p*K)x1 array of the Y_t values in companion form

%% Generate companion forms

% The p*K (column dim) of the PI array
pK = size(PI,2);

%Companion form of the PI array
Comp_Pi = [PI;eye(pK)];
Comp_Pi = Comp_Pi(1:pK,:);
 
if nargin>1
    
    %Companion form of the Constant array
    Comp_C = [C;zeros(pK,1)];
    Comp_C = Comp_C(1:pK,:);

end

if nargin>2
   
    % Reshape Y into companion form
    Comp_Y = reshape(flipud(Y_t)',1,[]);

end

end

