function save_EP_dog_NM_fits(k)
% function save_EP_H_NM_fits(k)
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

n_subs = 100;
delays = [0, 1, 3, 6, 10];
n_delays = length(delays);
s = mod(floor((k - 1) / n_delays), n_subs) + 1;
d = mod(k - 1, n_delays) + 1;

data_dir = '~/dopa_net/behavioral_experiments/psychtoolbox/data/';
load(strcat(data_dir, sprintf('s%03d_%02d.mat', s, delays(d))));
load(strcat(data_dir, sprintf('s%03d_%02d_d_stim.mat', s, delays(d))));
disp(strcat(['Loaded ', data_dir, sprintf('s%03d_%02d.mat', s, delays(d))]))

eval(sprintf('data = s%03d_%02d;', s, delays(d)))
eval(sprintf('d_stim = s%03d_%02d_d_stim;', s, delays(d)))
all_data(:, 1) = data;
all_data(:, 2) = d_stim;

[params_hat, params_0, log_like] = fit_EP_dog_NM(all_data);
n_params = length(params_hat);
aic = -2 * log_like + 2 * n_params;

% Determine which attempt this is.
attempt = 1;
while 1
    f_name = sprintf('s%03d_%02d_EP_dog_NM_%02d.mat', s, delays(d), attempt);
    full_f_name = strcat(data_dir, f_name);
    if ~(exist(full_f_name, 'file') == 2)
        save(full_f_name, 'params_hat', 'params_0', 'aic');
        break
    end
    attempt = attempt + 1;
end