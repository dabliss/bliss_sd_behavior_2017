function [params_hat, params_0, log_like] = fit_VMRW_dog(data)
% function [params_hat, params_0, log_like] = fit_VMRW_attraction(data)
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
%   kappa, xi, c, and s for best fit to data.
%
% params_0 : array
%   Initial guess for kappa, xi, c, and s.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors, d_stim)
        
        kappa = params(1);
        xi = params(2);
        a = params(3);
        w = params(4);
        
        min_a = -pi;
        max_a = pi;
        min_w = 0.4;
        max_w = 4.0;
        if any(kappa < 0) || any(xi < 0) || any(a < min_a) || any(a > max_a) || any(w < min_w) || any(w > max_w)
            log_like = -Inf;
            return
        end
        
        p_error = VMRW_dog(errors, kappa, xi, a, w, d_stim);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

params_0 = [rand * (70 - 0.6) + 0.6, ...  % kappa
    rand * (700 - 4) + 4, ...  % xi
    rand, ...  % a
    rand];  % w
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    data(:, 1), data(:, 2)), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end