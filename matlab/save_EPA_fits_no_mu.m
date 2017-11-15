function save_EPA_fits_no_mu(k)
% function save_EPA_fits(k)
%
% Parameters
% ----------
% k : integer
%   Identifier for the dataset to use.

if k < 1 || rem(k, 1)
    error('Invalid value for k supplied.')
end

% Seed the random number generator with the current time.
rng('shuffle');

% Use k to compute a subject (1-3) and delay (1-5) to fit.
n_subs = 100;
delays = [0, 1, 3, 6, 10];
n_delays = 5;
s = mod(floor((k - 1) / n_delays), n_subs) + 1;
d = mod(k - 1, n_delays) + 1;

% Load data for this subject and delay.
data_dir = '~/dopa_net/behavioral_experiments/psychtoolbox/data/';
load(strcat(data_dir, sprintf('s%03d_%02d.mat', s, delays(d))));

disp(strcat(['Loaded ', data_dir, sprintf('s%03d_%02d.mat', s, delays(d))]))

theta = linspace(-pi, pi, 1000);
eval(sprintf('data = s%03d_%02d;', s, delays(d)))
[params_hat, params_0, log_like] = fit_EPA_no_mu(data);
n_params = length(params_hat);
aic = -2 * log_like + 2 * n_params;
bic = -2 * log_like + log(length(data)) * n_params;
pdf = von_mises(theta, 0, params_hat(1));

% Determine which attempt this is.
attempt = 1;
while 1
    f_name = sprintf('s%03d_%02d_EPA_no_mu_%02d.mat', s, delays(d), attempt);  
    full_f_name = strcat(data_dir, f_name);
    if ~(exist(full_f_name, 'file') == 2)
        save(full_f_name, 'params_hat', 'theta', 'pdf', 'params_0', ...
            'aic', 'bic');
        break
    end
    attempt = attempt + 1;
end