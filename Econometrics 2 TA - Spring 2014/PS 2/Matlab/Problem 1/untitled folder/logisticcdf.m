function [ prob ] = logisticcdf( x )
prob=1./(1+exp(-x));
end