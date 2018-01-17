function ang = pix2angle(dist, width_cm, width_pix, pix)
%angle = pix2angle(display, pix)
%
% Converts monitor pixels into degrees of visual angle.
%
% Inputs:
% dist (distance from screen (cm))
% width_cm (width of screen (cm))
% width_pix (number of pixels of display in horizontal direction)
% pix (number of pixels to convert to an angle)
%
% Warning: assumes isotropic (square) pixels

% Written 11/1/07 gmb zre

% Calculate pixel size in cm.
pixSize = width_cm / width_pix;   % cm / pix

x = pix * pixSize;  % cm
rad = 2 * atan(x / (2 * dist));
ang = rad * 360 / (2 * pi);