function position = get_stim_rect(x_cent, y_cent, ecc_pix, ang, half_width)
% Get the position of the stimulus rectangle.
% 
% Arguments
% ---------
% x_cent : integer
%   Center pixel in the horizontal direction.
%
% y_cent : integer
%   Center pixel in the vertical direction.
%
% ecc_pix : number
%   Eccentricity of the stimulus from the center in pixels.
%
% ang : number
%   Angle of the stimulus in degrees.
%
% half_width : number
%   Half the width of the stimulus rectangle in pixels.

% Convert ang to radians.
rad = ang * 2 * pi / 360;

% Break ecc_pix into horizontal and vertical components using rad.
x_pix = ecc_pix * cos(rad);
y_pix = ecc_pix * sin(rad);

position = [x_cent + x_pix - half_width; ...
            y_cent + y_pix - half_width; ...
            x_cent + x_pix + half_width; ...
            y_cent + y_pix + half_width];