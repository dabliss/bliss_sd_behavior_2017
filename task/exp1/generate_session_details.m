function generate_session_details(session)
% function generate_session_details(session)
% 
% session_details contains, for each trial:
%
% stim_ang               -- angle for stimulus (random)
%
% delay                  -- 0, 1, 3, 6, or 10 seconds 

if exist(['session_details_' session '.mat'], 'file')
    disp('Session details already exist.')
    return
end

rng('shuffle');

% Demand that each delay length be equally represented among the trials.
delays = [zeros(1, 5) ones(1, 5) 3 * ones(1, 5) 6 * ones(1, 5) ...
          10 * ones(1, 5)];
n_trials = length(delays);
delays = delays(randperm(n_trials));

% Initialize container of session details.
session_details(n_trials).stim_ang = 0;
session_details(n_trials).delay = 0;

% Fill session_details.
for i = 1:n_trials
    session_details(i).stim_ang = rand * 360;
    session_details(i).delay = delays(i);
end

save(['session_details_' session], 'session_details')