function p = VMRW_swap(x, mu, k, xi, p_mem, d_stim)
% function p = VMRW_swap(x, mu, k, p_mem)
%
% Parameters
% ----------
% x : array
%   Error for each trial of one condition of a continuous-report task in
%   radians.
%
% mu : number
%   Systematic bias in radians.
%
% k : number
%   Concentration parameter.
%
% xi : number
%   Mean of Poisson distribution.
%
% p_mem : number
%   Probability with which item is in mind at the time of the response.
% 
% Returns
% -------
% p : array
%   Probability associated with each observed error.

[n_x, n_y] = size(d_stim);
p = zeros(n_x, n_y);
for trial_i = 1:n_x
    for trial_j = 1:n_y
        in_mind = p_mem * JN14_pdf_mu(x(trial_i, trial_j), mu, k, xi);
        swap = (1 - p_mem) * JN14_pdf_mu(x(trial_i, trial_j), ...
            mu + d_stim(trial_i, trial_j), k, xi);
        p(trial_i, trial_j) = in_mind + swap;
    end
end
