import pandas as pd
import numpy as np
from scipy import io as sio


def get_subject_data(subject, sessions, task, keys, indices,
                     n_trials_per_session):

    """Load a subject's data.

    Parameters
    ----------
    subject : integer
      Subject number.

    sessions : sequence or iterator
      Numbers for the sessions.

    task : string
      Directory with the data for the task being processed.

    keys : sequence
      Column headers from session_details for the DataFrame to be
      returned.

    indices : sequence
      Indices in session_details corresponding to the keys.

    n_trials_per_session : integer
      Number of trials per session.

    Returns
    -------
    all_sessions : pandas.DataFrame
      Data for all of a subject's sessions.
    
    """

    data_dir = '/home/despo/dbliss/dopa_net/behavioral_experiments/psychtoolbox/%s/data/' % task

    # Load the response data for the subject.
    for session in sessions:
        results_file = '%s%03d_%03d_results.txt' % (data_dir, subject, session)
        exec 'session_%03d = pd.read_csv("%s", sep="\t")' % (session,
                                                             results_file)
    all_sessions_response = pd.concat([eval('session_%03d' % session) for
                                       session in sessions], 
                                      ignore_index=True)

    # Load the presentation data for the subject.
    for session in sessions:
        session_file = '%ssession_details_%03d_%03d.mat' % (data_dir, subject,
                                                            session)
        exec 'session_%03d = sio.loadmat("%s")' % (session, session_file)
        session_slice = 'session_%03d["session_details"][0, :]' % session
        data_dict = {}
        for k, i in zip(keys, indices):
            command = 'np.squeeze(zip(*%s)[%d])' % (session_slice, i)
            exec '%s = %s' % (k, command)
            data_dict[k] = eval(k)
        exec 'session_%03d = pd.DataFrame(data_dict)' % (session,)
    all_sessions_presentation = pd.concat([eval('session_%03d' % session) for
                                           session in sessions],
                                          ignore_index=True)

    # Combine the response and presentation data.
    all_sessions = pd.concat([all_sessions_response,
                              all_sessions_presentation], axis=1)

    # Fix the response as needed.
    all_sessions.loc[all_sessions.response_angle < 0, 'response_angle'] += 360

    # Add a column for errors.
    if n_stimuli_per_trial == 1:
        all_sessions['errors'] = (all_sessions.response_angle -
                                  all_sessions.stimulus_angles)
    elif n_stimuli_per_trial == 2:
        errors = np.empty_like(all_sessions.response_angle)
        for i in range(len(errors)):
            errors[i] = (all_sessions.response_angle[i] -
                         eval('all_sessions.stimulus_angle_%d[i]' %
                              all_sessions.cue[i]))
        all_sessions['errors'] = errors
        
    # Correct case that difference is less than -180.
    all_sessions.loc[all_sessions.errors < -180, 'errors'] += 360
    # Correct case that difference is greater than 180.
    all_sessions.loc[all_sessions.errors >= 180, 'errors'] -= 360

    return all_sessions
