function vswm_task(session)
% function vswm_task(session)

% Open results file or return with error.
if exist([session '_results.txt'], 'file')
    disp('Results file already exists.')
    return
else
    results_file = fopen([session '_results.txt'], 'w');
end

% Write headers to file.
fprintf(results_file, '%s\t%s\t%s\n', 'response_time', ...
        'response_angle', 'response_eccentricity');

% Establish global display variables.
dist = 60;  % cm, distance from subject to screen.
width_cm = 50.917;  % width of the screen.
[width_pix, height] = Screen('WindowSize', 0);
x_cent = width_pix / 2;
y_cent = height / 2;

% Define the fixation square.
fix_square_half_ang = 0.25;  % in visual angle.
fix_square_half_pix = angle2pix(dist, width_cm, width_pix, ...
                                fix_square_half_ang);
fixation_square = [x_cent - fix_square_half_pix ...
                   y_cent - fix_square_half_pix ...
                   x_cent + fix_square_half_pix ...
                   y_cent + fix_square_half_pix];
               
% Define the diameter of each stimulus in pixels.
stim_diam_ang = 1;  % in visual angle.
stim_diam_pix = angle2pix(dist, width_cm, width_pix, stim_diam_ang);
               
% Set the time for each task stage (except for the delay, which is
% trial-specific).  All times are in seconds.
fix_time = 1;
stim_time = 1;

% Define the area within which mouse clicks count as responses.
stim_eccentricity = 12;  % in visual angle.
stim_ecc_pix = angle2pix(dist, width_cm, width_pix, stim_eccentricity);

% Load the trial-specific variables.
load(['session_details_' session]);
n_trials = length(session_details);

% Compute the stimulus positions from the angles supplied in
% session_details.
stim_positions = zeros(n_trials, 4);
for i = 1:n_trials
    stim_positions(i, :) = get_stim_rect(x_cent, y_cent, stim_ecc_pix, ...
                                         session_details(i).stim_ang, ...
                                         stim_diam_pix / 2);
end

% Two colors will be used: gray for the background and black for the
% fixation square and stimuli.
gray = GrayIndex(0);
black = BlackIndex(0);

% Set the screen gray.
window = Screen('OpenWindow', 0, gray);

% Avoid font error.
%Screen('TextFont', window, ...
%       '-adobe-times-medium-i-normal--34-240-100-100-p-168-iso10646-1');
HideCursor;
WaitSecs(1);

% Flip up the welcome screen.
Screen('TextSize', window, 40);
DrawFormattedText(window, ...
    'Click the mouse when you are ready to begin.', 'center', 'center');
Screen('Flip', window);
GetClicks(0);

% Do the countdown.
Screen('TextSize', window, 40);
DrawFormattedText(window, 'First trial begins in 5.', 'center', 'center');
[~, in_5] = Screen('Flip', window);
Screen('TextSize', window, 40);
DrawFormattedText(window, 'First trial begins in 4.', 'center', 'center');
[~, in_4] = Screen('Flip', window, in_5 + 1);
Screen('TextSize', window, 40);
DrawFormattedText(window, 'First trial begins in 3.', 'center', 'center');
[~, in_3] = Screen('Flip', window, in_4 + 1);
Screen('TextSize', window, 40);
DrawFormattedText(window, 'First trial begins in 2.', 'center', 'center');
[~, in_2] = Screen('Flip', window, in_3 + 1);
Screen('TextSize', window, 40);
DrawFormattedText(window, 'First trial begins in 1.', 'center', 'center');
[~, in_1] = Screen('Flip', window, in_2 + 1);

% Start the task.
for i = 1:n_trials
    
    % Flip up the fixation screen.
    Screen('FillRect', window, black, fixation_square);
    if i == 1
        [~, fix_start] = Screen('Flip', window, in_1 + 1);
    else
        [~, fix_start] = Screen('Flip', window);
    end
    
    % Flip up the stimulus.
    Screen('FillRect', window, black, fixation_square);
    Screen('FillOval', window, black, stim_positions(i, :), stim_diam_pix);
    [~, stim_start] = Screen('Flip', window, fix_start + fix_time);

    % Flip up the delay.
    Screen('FillRect', window, black, fixation_square);
    [~, delay_start] = Screen('Flip', window, stim_start + stim_time);

    % Flip up the response screen.
    SetMouse(x_cent, y_cent);
    [~, probe_start] = Screen('Flip', window, ...
                              delay_start + session_details(i).delay);
    
    % Record the first mouse click that's within an annulus that spans all
    % possible pixels covered by stimuli.
    ShowCursor('Arrow');
    while 1
        [response_x, response_y, buttons] = GetMouse;
        response_time = GetSecs - probe_start;
        if any(buttons)
            [resp_rad, resp_ecc] = cart2pol(response_x - x_cent, ...
                                            response_y - y_cent);
            resp_ang = resp_rad / (2 * pi) * 360;
            if resp_ang < 0
                resp_ang = resp_ang + 360;
            end
            break
        end
    end
    HideCursor;
    
    % Convert the response eccentricity from pixels to degrees of visual 
    % angle so that interpretation of the results file is not dependent on 
    % knowledge of which computer was used.
    response_ecc = pix2angle(dist, width_cm, width_pix, resp_ecc);
    
    % Write to the results file.
    fprintf(results_file, '%f\t%f\t%f\n', response_time, resp_ang, ...
            response_ecc);
        
end

Screen('CloseAll');
ShowCursor;