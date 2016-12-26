function [ rho1AR1, rho1AR2, rho2AR2, rho1AR1IV, rho1AR2IV, rho2AR2IV, rho1GMM ] = MC( N,Reps,DGP,lags,moment,momentmc,e )

% This function will conduct the Monte-Carlo simulation for our problem.

rho1AR1=zeros(Reps-lags,1);

rho1AR2=zeros(Reps-lags,1);
rho2AR2=zeros(Reps-lags,1);

rho1AR1IV=zeros(Reps-lags,1);

rho1AR2IV=zeros(Reps-lags,1);
rho2AR2IV=zeros(Reps-lags,1);

rho1GMM=zeros(Reps-lags,1);

for i=1:Reps
    %% Simulation of the data 
    
    x=randn(N,1);          
    ehat=randn(N,1);            
    
    y=zeros(N,1);
   
    for j=3:N;
        y(j,1)=DGP(1)+DGP(2)*y(j-1)+DGP(3)*y(j-2)+DGP(4)*x(j)+ehat(j);
    end
    
    %% Extract the dependant variable and lags for the process
    
    Yt=lagmatrix(y,0:lags);
    Y=Yt(lags+1:end,1);
    Yt1=Yt(lags+1:end,2);
    Yt2=Yt(lags+1:end,3);
    
%     Y=y(3:end);
%     Yt1=y(2:end-1);
%     Yt2=y(1:end-2);
    
    %% Lag the exogenous variables for IV estimation later
    
    Xt=lagmatrix(x,0:lags);
    x=Xt(lags+1:end,1);
    xt1=Xt(lags+1:end,2);        % for the AR2 instrument
    xt2=Xt(lags+1:end,3);
    
    
%     xt2=[randn(2,1);x(1:end-2)];
%     xt1=[xt2(2:end);x(end-1)];
    
    %% AR1 and AR2 Data Matrices
    
    X1=[ones(N-lags,1) Yt1 x];
    X2=[ones(N-lags,1) Yt1 Yt2 x];
    
    %% Simulation of the OLS AR1 process
    
    AR1param=(X1'*X1)\X1'*Y;
    
    rho1AR1(i)=AR1param(2)-DGP(2);
    
    %% Simulation of the OLS AR2 process
    
    AR2param=(X2'*X2)\X2'*Y;
    
    rho1AR2(i)=AR2param(2)-DGP(2);
    rho2AR2(i)=AR2param(3)-DGP(3);
    
    %% Simulation of the IV AR1 process
    
    Z1=[ones(N-lags,1) x xt1];

    AR1IVparam=(Z1'*X1)\Z1'*Y;

    rho1AR1IV(i)=AR1IVparam(2)-DGP(2);
    
    %% Simulation of the IV AR2 process
    
    Z2=[ones(N-lags,1) x xt1 xt2];

    AR2IVparam=(Z2'*X2)\Z2'*Y;

    rho1AR2IV(i)=AR2IVparam(2)-DGP(2);
    rho2AR2IV(i)=AR2IVparam(3)-DGP(3);
    
    %% GMM of the the IV AR1 process
    
    Zgmm=[Z1 xt1.^2];% x.^2 x.^3 xt1.^3 (xt1.^2+x.^4) (x.^2+xt1.^4) (x+xt1.^5) (x.^5+xt1)];

    Gmmdelta=GMM(AR1param,Y,X1,Zgmm,moment,momentmc,e);
    
    rho1GMM(i)=Gmmdelta(2)-DGP(2);
   
end

end

