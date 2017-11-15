function [params_hat, params_0, log_like] = fit_VPA_no_mu(errors)
% function [params_hat, params_0, log_like] = fit_VPA(errors)
%
% Fit the model introduced in van den Berg et al., '12, PNAS and Fougnie et
% al., '12, Nat Commun to data using MLE.
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
%   mu, k_bar, and tau for best fit to data.
%
% params_0 : array
%   Initial guess for mu, k_bar, and tau.
%
% log_like : float
%   Log likelihood of the final model fit.

    % Define the function to be minimized.
    function log_like = get_log_like(params, errors)
        
        j_bar = params(1);
        tau = params(2);
        
        % Return log_like = -Inf if k_bar or tau is negative.
        if j_bar < 0 || tau < 0
            log_like = -Inf;
            return
        end
        
        p_error = VPA(errors, 0, j_bar, tau, all_kappa, all_J);
        
        % Set all p < 0.00001 to p = 0.00001 to avoid -Inf in the sum
        % below.
        p_error = max(p_error, 1e-5);
                
        log_like = sum(log(p_error));
        
    end

all_kappa = [linspace(0, 10, 250) linspace(10.001, 1e4, 250)];
all_J = all_kappa .* besseli(1, all_kappa, 1) ./ besseli(0, all_kappa, 1);

params_0 = [rand * (500 - 20) + 20, ...  % j_bar
    rand * (50 - 0) + 0];  % tau
[params_hat, neg_log_like] = fminsearch(@(pars) -get_log_like(pars, ...
    errors), params_0, optimset('Display', 'iter'));
log_like = -neg_log_like;
end