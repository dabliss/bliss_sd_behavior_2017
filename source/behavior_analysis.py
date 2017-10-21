import pandas as pd
import numpy as np
from scipy import io as sio
import matlab
import matlab.engine
from scipy.optimize import least_squares


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


def cut_bad_trials(df, iti=False):
    """Cut trials where response was near origin."""
    mean = df.errors.mean()
    std = df.errors.std()
    bad_ind = df.index[(df.response_eccentricity < 5) |
                       (df.errors > mean + std * 3) |
                       (df.errors < mean - std * 3)]
    after_bad_ind = bad_ind + 1
    if not iti:
        df.loc[after_bad_ind, ['d_stim', 'prev_delay', 'prev_stim']] = np.nan
    else:
        if after_bad_ind[-1] == 1008:
            after_bad_ind = after_bad_ind[:-1]
        df.loc[after_bad_ind, ['d_stim', 'prev_stim']] = np.nan
    return df.drop(bad_ind)


def get_sys_error(df):
    """Get the systematic error as a function of stimulus location.

    Add 'global_sys_error', 'resid_error', 'prev_stim', and 'd_stim' to
    df.

    Parameters
    ----------
    df : pandas.DataFrame
      Data Frame for a subject.

    Returns
    -------
    df : pandas.DataFrame
      Updated Data Frame.
    
    """
    eng = matlab.engine.start_matlab()
    stimulus_angles = df['stimulus_angles']
    errors = df['errors']
    n_trials = len(stimulus_angles)
    x = matlab.double(list(np.array([stimulus_angles,
                                     stimulus_angles + 360,
                                     stimulus_angles + 720]).flatten()))
    y = matlab.double(list(np.tile(errors, 3)))
    smoothed = eng.smooth(x, y, 155, 'loess')
    sys_error = np.squeeze(np.array(smoothed)[n_trials:n_trials * 2])
    df['global_sys_error'] = sys_error
    df['global_resid_error'] = df['errors'] - df['global_sys_error']
    return df


def clifford(x, c, s):
    """Compute Clifford's tilt model function.

    Parameters
    ----------
    x : numpy.array
      Location of each trial's previous target, relative to the current
      target in radians.

    c : number
      c parameter.

    s : number
      s parameter.

    Returns
    -------
    numpy.array
      The value of Clifford's function for the submitted values of x.
    
    """
    theta_ad = np.arcsin((np.sin(x)) / np.sqrt(((s * np.cos(x) - c)) ** 2 +
                                               (np.sin(x)) ** 2))
    test = s * np.cos(x) - c < 0
    theta_ad[test] = np.pi - theta_ad[test]
    return np.mod(theta_ad - x + np.pi, 2 * np.pi) - np.pi


def fit_clifford(y, x):

    """Fit Clifford's tilt model to observed errors.

    Parameters
    ----------
    y : numpy.array
      Residual error (in radians).

    x : numpy.array
      Location of previous stimulus relative to the current one (in
      radians).
    
    Returns
    -------
    float
      Computed c for the Clifford fit.

    float
      Computed s for the Clifford fit.

    float
      Computed sign for the Clifford fit.

    float
      Cost for the fit.

    """

    def _solver(params):
        c, s, m = params
        m = np.sign(m)
        return y - m * clifford(x, c, s)

    min_c = 0.0
    max_c = 1.0
    
    min_s = 0.0
    max_s = 1.0

    min_m = -1.0
    max_m = 1.0
    
    min_cost = np.inf
    for _ in range(200):
        params_0 = [np.random.rand() * (max_c - min_c) + min_c,
                    np.random.rand() * (max_s - min_s) + min_s,
                    np.random.rand() * (max_m - min_m) + min_m]
        try:
            result = least_squares(_solver, params_0,
                                   bounds=([min_c, min_s, min_m],
                                           [max_c, max_s, max_m]))
        except ValueError:
            continue
        if result['cost'] < min_cost:
            best_params, min_cost = result['x'], result['cost']
    try:
        return best_params[0], best_params[1], np.sign(best_params[2]), min_cost
    except UnboundLocalError:
        return np.nan, np.nan, np.nan, min_cost

    
def dog(x, a, w):
    c = np.sqrt(2) / np.exp(-0.5)
    return x * a * w * c * np.exp(-(w * x) ** 2)


def fit_dog(y, x):

    def _solver(params):
        a, w = params
        return y - dog(x, a, w)

    min_a = -np.pi
    max_a = np.pi

    min_w = 0.4
    max_w = 4.0

    min_cost = np.inf
    for _ in range(200):
        params_0 = [np.random.rand() * (max_a - min_a) + min_a,
                    np.random.rand() * (max_w - min_w) + min_w]
        try:
            result = least_squares(_solver, params_0,
                                   bounds=([min_a, min_w],
                                           [max_a, max_w]))
        except ValueError:
            continue
        if result['cost'] < min_cost:
            best_params, min_cost = result['x'], result['cost']
    try:
        return best_params[0], best_params[1], min_cost
    except UnboundLocalError:
        return np.nan, np.nan, min_cost

    
def print_fit_goodness(data_frames):

    n_subs = len(data_frames)
    aic_c_values = np.empty((2, n_subs))
    for i, df in enumerate(data_frames):

        d_stim = np.array(df['d_stim'])
        error = np.array(df['global_resid_error'])

        ind = ~np.isnan(d_stim)
        d_stim_rad = np.deg2rad(d_stim[ind])
        error_rad = np.deg2rad(error[ind])
            
        assert len(d_stim_rad) == len(error_rad)
        total_n = len(d_stim_rad)

        # Get Clifford fit.
        cliff_params = fit_clifford(error_rad, d_stim_rad)

        # Compute AICc.
        rss = ((error_rad -
                cliff_params[2] * clifford(d_stim_rad, cliff_params[0],
                                           cliff_params[1])) ** 2).sum()
        k = len(cliff_params[:-1])
        aic = 2 * k + total_n * np.log(rss)
        aic_c_values[0, i] = aic + 2 * k * (k + 1) / (total_n - k - 1)

        # Get DoG fit.
        dog_params = fit_dog(error_rad, d_stim_rad)

        # Compute AICc.
        rss = ((error_rad - dog(d_stim_rad, dog_params[0],
                                dog_params[1])) ** 2).sum()
        k = len(dog_params[:-1])
        aic = 2 * k + total_n * np.log(rss)
        aic_c_values[1, i] = aic + 2 * k * (k + 1) / (total_n - k - 1)

    print 'Clifford - DoG:'
    print (aic_c_values[0, :] - aic_c_values[1, :]).mean()
    print (aic_c_values[0, :] - aic_c_values[1, :]).std() / np.sqrt(n_subs)
