% Alberto Ramirez
% Problem 5 Solution Code

clc
clear all

% Set the Random Number Generators all to default start values

%% Define the variables

var.n=1000;
var.sim=1000;

%% Monte-Carlo Simulation

[MC.R2,MC.t,MC.StdErr,MC.test,MC.beta,MC.plbq,MC.p]=MCRW(var.n,var.sim);

% p-value method of rejecting the null

for i=1:var.n
    if MC.p(i)<0.05         % We reject the null
       MC.pcount(i)=1;
    end
end

%% Output the results
clc

var.index=ceil(rand*var.n);

fprintf('%6s %1.0f \n\n','Realization from one time series, simulation #',var.index);
fprintf('R-squared      = %2.3f \n',MC.R2(var.index));
% fprintf('Rbar-squared   = %12.4f \n',ols.rbar);
fprintf('LBQ p-value    = %2.3f \n\n',MC.plbq(var.index));
if MC.plbq(var.index) < 0.05;
    fprintf('LBQ Test Result: Reject null => "There is AC in error term". \n \n')
else
    fprintf('LBQ Test Result: Fail to reject null => "There is no AC in error term". \n \n');
end
fprintf('*****************************************************************\n');

char.namestr = ' Variable';
char.bstring = '    Coef.';
char.sdstring= 'Std. Err.';
char.tstring = '  t-stat.';
char.pstring = '  p-value';

fprintf('\n')
fprintf('%12s %12s %12s %12s %12s \n',char.namestr, ...
    char.bstring,char.sdstring,char.tstring,char.pstring)
fprintf('%12s %12.3f %12.3f %12.3f %12.3f \n',...
    '    Constant',...
    MC.beta(var.index,1),MC.StdErr(var.index,1),MC.t(var.index,1),MC.p(var.index,1))
fprintf('%12s %12.3f %12.3f %12.3f %12.3f \n\n',...
    '        Beta',...
    MC.beta(var.index,2),MC.StdErr(var.index,2),MC.t(var.index,2),MC.p(var.index,2))
fprintf('*****************************************************************\n \n');

fprintf('Test the following hypothesis: \n\n')
fprintf('Ho: Beta = 0 \n')
fprintf('H1: Beta not = 0 \n \n')

if abs(MC.t(var.index,2)) > 1.96
    fprintf('Reject the null: Beta is significantly different from 0. \n \n')
else
    fprintf('Fail to reject null: Beta is not significantly different from 0. \n \n')
end
fprintf('*****************************************************************\n \n');
fprintf('Results from the Monte Carlo Simulation \n\n')
fprintf('%6s %0.3f %1s %0.3f \n\n','The mean of beta is',mean(MC.beta(:,2)),' with a standard error of',std(MC.beta(:,2)));
fprintf('%6s %0.3f %1s %0.3f \n\n','The mean of our t-statistic is',mean(MC.t(:,2)),' with a standard deviation of',std(MC.t(:,2)));
fprintf('%6s %0.3f \n\n','The median of the standard error of beta is',median(MC.StdErr(:,2)));
fprintf('%6s %0.3f %6s %0.3f \n\n','The median of Rsq is',median(MC.R2),'with a mean of',mean(MC.R2));
fprintf('%12s %6.2f %1s\n','MC Result: Percentage of Times Fail to Reject null: ',(1-mean(MC.test))*100,'%');
fprintf('%12s %12.2f %1s\n\n','MC Result: Percentage of Times Null Rejected: ',mean(MC.test)*100,'%');
fprintf('%6s %0.3f %1s %0.3f\n\n','The average size of the test is',mean(MC.p(:,2)),'and the median size of the test is',median(MC.p(:,2)));


%% Display Plots

disp.figure=figure('Units','normalize','Position',[0 0 1 1]); orient landscape;
subplot(2,2,1); hist(MC.beta(:,2),25)
title('Spurious     Regression     Beta     Distribution');
subplot(2,2,2); hist(MC.StdErr(:,2),25)
title('Spurious     Regression     StdErr    Distribution');
subplot(2,2,3); hist(MC.R2,25)
title('Spurious     Regression     R-Square     Distribution');
subplot(2,2,4); histfit(MC.t(:,2),25,'t'); hold on
histfit(randn(1000,1).*std(MC.t(:,2)),25,'normal');
title('Spurious     Regression     t-stat     Distribution');










