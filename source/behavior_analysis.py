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
    all_sessions['errors'] = (all_sessions.response_angle -
                              all_sessions.stimulus_angles)
        
    # Correct case that difference is less than -180.
    all_sessions.loc[all_sessions.errors < -180, 'errors'] += 360
    # Correct case that difference is greater than 180.
    all_sessions.loc[all_sessions.errors >= 180, 'errors'] -= 360

    return all_sessions


def add_columns(df):
    """Add columns to df.

    Parameters
    ----------
    df : pandas.DataFrame
      Data Frame for a subject.

    Returns
    -------
    df : pandas.DataFrame
      Updated Data Frame.
    
    """
    key = 'stimulus_angles'
    prev_stim = np.insert(np.array(df[key])[:-1], 0, np.nan)
    future_stim = np.insert(np.array(df[key])[1:], len(df[key]) - 1,
                            np.nan)
    prev_delay = np.insert(np.array(df.delays, dtype=float)[:-1], 0,
                           np.nan)
    n_trials_per_session = 25
    # The first trial of each session (not just the overall first
    # trial) has no previous stimulus.
    prev_stim[::n_trials_per_session] = np.nan
    prev_delay[::n_trials_per_session] = np.nan
    # The last trial of each session (not just the overall
    # last trial) has no future stimulus.
    future_stim[n_trials_per_session-1::n_trials_per_session] = np.nan
    df['prev_stim'] = prev_stim
    df['prev_delay'] = prev_delay
    df['future_stim'] = future_stim
    d_stim = prev_stim - np.array(df[key])
    d_stim_future = future_stim - np.array(df[key])
    # Correct case that difference is less than -180.
    mask = ~np.isnan(d_stim)
    mask[mask] &= d_stim[mask] < -180
    d_stim[mask] += 360
    # Correct case that difference is greater than 180.
    mask = ~np.isnan(d_stim)
    mask[mask] &= d_stim[mask] >= 180
    d_stim[mask] -= 360
    # Make the same corrections for d_stim_future.
    mask = ~np.isnan(d_stim_future)
    mask[mask] &= d_stim_future[mask] < -180
    d_stim_future[mask] += 360
    mask = ~np.isnan(d_stim_future)
    mask[mask] &= d_stim_future[mask] >= 180
    d_stim_future[mask] -= 360
    # Add columns to df.
    df['d_stim'] = d_stim
    df['d_stim_future'] = d_stim_future
    return df
