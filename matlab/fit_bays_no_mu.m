function [params_hat, params_0, log_like] = fit_bays_no_mu(errors)
% function [params_hat, params_0, log_like] = fit_bays_no_mu(errors)
%
% Fit the model introduced in Bays, 2014, J Neurosci and extended in Bays,
% 2016, submitted to data using MLE.
%
% Parameters
% ----------
% errors : array
%   Error for each trial of one condition of a continuous-report task in
%   radians.
% 
% Returns
% -------
% params_hat : array
%   mu, kappa, and xi for best fit to data.
%
% params_0 : array
%   Initial guess for mu, kappa, and xi.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors)
        
        kappa = params(1);
        xi = params(2);
        
        % Return log_like = -Inf if kappa or xi is negative.
        if kappa < 0 || xi < 0
            log_like = -Inf;
            return
        end
        
        p_error = poissvmwalkpdf(errors, 0, kappa, xi);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

params_0 = [rand * (70 - 0.6) + 0.6, ...  % kappa
    rand * (700 - 4) + 4];  % xi
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    errors), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end