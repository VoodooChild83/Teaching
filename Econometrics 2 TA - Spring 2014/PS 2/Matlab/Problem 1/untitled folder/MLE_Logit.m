function [ log_like ] = MLE_Logit(l,X,Y)
q=2*Y-1;
F=logisticcdf(q.*(X*l));
log_like=-sum(log(F));
end

