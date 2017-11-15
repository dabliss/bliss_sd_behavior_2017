function p = von_mises(x, mu, k)
% function p = von_mises(x, mu, k)
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
% k : number
%   Concentration parameter.
%
% Returns
% -------
% p : array
%   Probability associated with each observed error.

p = 1 / (2 * pi * besseli(0, k)) * exp(k * cos(x - mu));