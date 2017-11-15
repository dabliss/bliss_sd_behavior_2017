function p = VPA(x, mu, j_bar, tau, all_kappa, all_J)
% function p = VPA(x, mu, k_bar, tau)
%
% PDF of the model introduced in Zhang & Luck, '08, Nature.
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
% k_bar : number
%   Mean concentration parameter value.
%
% tau : number
%   Shape parameter for the Gamma distribution.  (Variance of the
%   distribution divided by k_bar.)
%
% Returns
% -------
% p : array
%   Probability associated with each observed error.

n_gamma_bins = 50;
X = linspace(0, 1, n_gamma_bins + 1);
X = X(2:end) - diff(X(1:2)) / 2;

j_vector = gaminv(X, j_bar / tau, tau);
k_vector = interp1(all_J, all_kappa, j_vector, 'linear', 'extrap');

k_vector = min(k_vector, 1e6);

p = 0;
for i = 1:n_gamma_bins
    p = p + von_mises(x, mu, k_vector(i));
end
p = p / n_gamma_bins;