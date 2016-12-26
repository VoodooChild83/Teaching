function [ c,ceq,gradc,gradceq ] = Constraint( nu,X,Y )
% constraint function with gradient

c = [];

ceq = sum(exp(X*nu(2:3))-nu(1));

gradc = [];

gradceq = -sum(Y/nu(1) - 1);

end

