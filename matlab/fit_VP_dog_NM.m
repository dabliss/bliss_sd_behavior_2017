function [params_hat, params_0, log_like] = fit_VP_dog_NM(data)
% function [params_hat, params_0, log_like] = fit_EP_H_NM(data)
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
%   kappa, h, and w for best fit to data.
%
% params_0 : array
%   Initial guess for kappa, h, and w.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors, d_stim)
        
        j_bar = params(1);
        tau = params(2);
        a = params(3);
        w = params(4);
        
        min_a = -pi;
        max_a = pi;
        min_w = 0.4;
        max_w = 4.0;
        if any(j_bar < 0) || any(tau < 0) || any(a < min_a) || any(a > max_a) || any(w < min_w) || any(w > max_w)
            log_like = -Inf;
            return
        end
        
        p_error = VP_dog_NM(errors, j_bar, tau, a, w, d_stim, all_kappa, all_J);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

all_kappa = [linspace(0, 10, 250) linspace(10.001, 1e4, 250)];
all_J = all_kappa .* besseli(1, all_kappa, 1) ./ besseli(0, all_kappa, 1);

params_0 = [rand * (500 - 20) + 20, ...  % j_bar
    rand * (50 - 0) + 0, ...  % tau
    rand, ...  % a
    rand];  % w
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    data(:, 1), data(:, 2)), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end