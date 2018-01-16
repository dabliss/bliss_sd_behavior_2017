function p = VMRW_dog(x, k, xi, a, w, d_stim)
% function p = VMRW_attraction(x, k, xi, c, s, d_stim)
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
% xi : number
%   Mean of the Poisson distribution.
%
% c : number
%   Centering parameter for the Clifford fit.
%
% s : number
%   Scaling parameter for the Clifford fit.
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
        p(trial_i, trial_j) = JN14_pdf_mu(x(trial_i, trial_j), ...
            b(trial_i, trial_j), k, xi);
    end
end