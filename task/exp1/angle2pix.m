function pix = angle2pix(dist, width_cm, width_pix, ang)
% pix = angle2pix(display,ang)
%
% converts visual angles in degrees to pixels.
%
% Inputs:
% dist (distance from screen (cm))
% width_cm (width of screen (cm))
% width_pix (number of pixels of display in horizontal direction)
% ang (visual angle in degrees)
%
% Warning: assumes isotropic (square) pixels

rad = ang * 2 * pi / 360;
x = 2 * dist * tan(rad / 2);

% Calculate pixel size
pixSize = width_cm / width_pix;   % cm / pix

pix = round(x/pixSize);   % pix