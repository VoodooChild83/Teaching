% create the function:

function [result,t_value,t_stat] = Sig_Test(beta_hat,beta_null,SE,alpha,SS)

    conf_level = 1 - alpha/2;
    
    t_value = (beta_hat - beta_null)./(SE);

    t_stat = tinv(conf_level, SS-1);
    
    if abs(t_value)>t_stat
        result = 'Reject the null in favor of alternative: parameters not equal';
    else
        result = 'Fail to reject the null in favor of alternative: parameters statistically equal';
    end

end