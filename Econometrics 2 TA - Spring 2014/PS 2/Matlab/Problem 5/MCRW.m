function [ rsqr,t,stderr,test,beta,plbq,pt ] = MCRW( n,sim )
%% Monte Carlo Simulation of the Random Walk Process

% initialize the variables:

rsqr=zeros(n,1);
t=zeros(n,2);
stderr=zeros(n,2);
test=zeros(n,1);
beta=zeros(n,2);
plbq=zeros(n,1);
pt=zeros(n,2);

rng('default');

for z=1:sim
    
    ux=randn(n,1);
    uy=randn(n,1);

%% Create the y variable:

   yt1=zeros(n,1);

   for i=1:n;
       yt1(i+1)=yt1(i)+uy(i);  % fill in the lags
   end

   y=yt1(2:n+1);               % take the forward and call it y

%% Create the x variable:

   xt1=zeros(n,1);

   for j=1:n;
       xt1(j+1)=xt1(j)+ux(j);
   end

   x=xt1(2:n+1);

%% Regress y upon x:

   X=[ones(n,1) x];

   beta(z,:)=((X'*X)\X'*y)';

%% Estimate Robust Var-Covar of estimate: Conduct D-W for detection of AC

   [Avar,~,plbq(z)]=HAC(beta(z,:)',y,X,X);
   
%% Generate Statistics

   % Standard Error
   stderr(z,:)=sqrt(diag(Avar))';

   % t-Statistic
   t(z,:)=beta(z,:) ./ stderr(z,:);

   % R^2 
   ym = y - mean(y);
   rsqr1 = (y-X*beta(z,:)')'*(y-X*beta(z,:)');
   rsqr2 = ym'*ym;
   rsqr(z) = 1 - rsqr1/rsqr2;                 % r-squared
   
   % Count the occuarnaces that the null is rejected (that is, we have that
   % beta is not equal to 0)
   
   if abs(t(z,2)) > 1.96
      test(z)=1;
   end 
   
   % As a second form, generate the p-values to get an estimate of the
   % size of the tests (multiply by 2 to get the two-sided p-values).
   % Counting the times this measure is below 0.05 gives us the same result
   % as the above. 
   
   pt(z,:)=2*tcdf(-abs(t(z,:)),n-2);
   
end


end

