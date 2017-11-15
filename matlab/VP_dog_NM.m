function p = VP_dog_NM(x, j_bar, tau, a, w, d_stim, all_kappa, all_J)
% function p = EP_with_history(x, k, h, w, d_stim)
%
% Return p(x|d_stim).
%
% Parameters
% ----------
% x : array
%   Error for each trial of one condition of a continuous-report task in
%   radians (negative is clockwise, positive is counter-clockwise).
%
% k : number
%   Concentration parameter.
%
% h : number
%   Height parameter for the Gabor fit.
%
% w : number
%   Width parameter for the Gabor fit.
%
% d_stim : array
%   Location of previous trial's stimulus relative to this trial's stimulus
%   in radians (negative means previous stimulus was clockwise relative to
%   this one, positive means previous stimulus was counter-clockwise
%   relative to this one).
% 
% Returns
% -------
% p : array
%   Probability associated with each observed error.

b = dog(d_stim, a, w);
% d_stim and x should be row vectors of the same size, but just in case
% matrices were passed in, split the number of trials into n_x and n_y --
% the number in either dimension of a potentially two-dimensional d_stim.
[n_x, n_y] = size(d_stim);
p = zeros(n_x, n_y);
for trial_i = 1:n_x
    for trial_j = 1:n_y
        p(trial_i, trial_j) = VPA(x(trial_i, trial_j), ...
            b(trial_i, trial_j), j_bar, tau, all_kappa, all_J);
    end
end