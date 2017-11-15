function [params_hat, params_0, log_like] = fit_VMRW_swap(data)
% function [params_hat, params_0, log_like] = fit_VMRW_swap(data)
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
%   kappa, xi, p_mem for best fit to data.
%
% params_0 : array
%   Initial guess for kappa, xi, and p_mem.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors, d_stim)
        
        kappa = params(1);
        xi = params(2);
        p_mem = params(3);
        
        if kappa < 0 || xi < 0 || p_mem < 0 || p_mem > 1
            log_like = -Inf;
            return
        end
        
        p_error = VMRW_swap(errors, 0, kappa, xi, p_mem, d_stim);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

params_0 = [rand * (70 - 0.6) + 0.6, ...  % kappa
    rand * (700 - 4) + 4, ...  % xi
    rand * (1 - 0.5) + 0.5];  % p_mem
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    data(:, 1), data(:, 2)), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end