function [params_hat, params_0, log_like] = fit_EPA_no_mu(errors)
% function [params_hat, params_0, log_like] = fit_EPA(errors)
%
% Fit the model introduced in Zhang & Luck, '08, Nature to data using MLE.
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
%   mu and kappa for best fit to data.
%
% params_0 : array
%   Initial guess for mu and kappa.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors)
        
        kappa = params(1);
        
        % Return log_like = -Inf if kappa is negative.
        if kappa < 0
            log_like = -Inf;
            return
        end
        
        % VDB says this is needed.
        kappa = min(kappa, 1e6);
        
        p_error = von_mises(errors, 0, kappa);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

params_0 = [rand * (500 - 20) + 20];  % kappa
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    errors), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end