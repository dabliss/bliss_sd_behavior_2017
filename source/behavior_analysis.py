import os

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


def save_for_permutation(data_frames, sub_nums, task_name='', perception=False,
                         only_delay=None, previous=False, future=False):

    """Save d_stim and global_resid_error for permutation tests.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    sub_nums : tuple
      Subject numbers

    task_name : string (optional)
      Specifier for the task.

    perception : boolean (optional)
      Whether to save the 0s delay (True) or all the other delays
      (False).

    only_delay : integer (optional)
      Only delay to use.

    previous : boolean (optional)
      Whether or not to use the previous trial's delay (as opposed to the
      current trial's).

    future : boolean (optional)
      Whether or not to use future_d_stim as d_stim.

    """
    
    package_dir = '/home/despo/dbliss/dopa_net/'
    data_dir = package_dir + 'behavioral_experiments/psychtoolbox/data/'

    if perception:
        per_string = '_perception'
    else:
        per_string = ''

    for i, (df, sub) in enumerate(zip(data_frames, sub_nums)):
        f_name = data_dir + 's%03d' % sub

        # Get d_stim_name.
        if not previous:
            try:
                d_stim_name = f_name + '_d_stim%s%s%02d.npy' % (task_name,
                                                                per_string,
                                                                only_delay)
            except TypeError:
                if perception:
                    d_stim_name = f_name + '_d_stim%s%s.npy' % (task_name,
                                                                per_string)
                else:
                    if not future:
                        d_stim_name = f_name + '_d_stim%s_all_delays.npy' % (
                            task_name,)
                    else:
                        d_stim_name = (f_name +
                                       '_d_stim%s_all_delays_future.npy' % (
                                task_name,))
        else:
            d_stim_name = f_name + '_d_stim%s%s%02d_previous.npy' % (
                task_name, per_string, only_delay)

        # Get d_stim.
        if perception:
            d_stim = np.array(df.loc[df.delays == 0, 'd_stim'])
        elif only_delay is None:
            if not future:
                d_stim = np.array(df['d_stim'])
            else:
                d_stim = np.array(df['d_stim_future'])
        else:
            if not previous:
                d_stim = np.array(df.loc[df.delays == only_delay, 'd_stim'])
            else:
                d_stim = np.array(df.loc[df.prev_delay == only_delay, 'd_stim'])
        ind = ~np.isnan(d_stim)
        d_stim_rad = np.deg2rad(d_stim[ind])

        # Get error_name.
        if not previous:
            try:
                error_name = (f_name + '_global_resid_error%s%s%02d.npy' % (
                    task_name, per_string, only_delay))
            except TypeError:
                if perception:
                    error_name = (f_name + '_global_resid_error%s%s.npy' % (
                        task_name, per_string))
                else:
                    if not future:
                        error_name = (f_name + '_global_resid_error' +
                                      '%s_all_delays.npy' % (task_name,))
                    else:
                        error_name = (f_name + '_global_resid_error' +
                                      '%s_all_delays_future.npy' %
                                      (task_name,))
        else:
            error_name = (f_name + '_global_resid_error%s%s%02d_previous.npy' %
                          (task_name, per_string, only_delay))

        # Get error.
        if not previous:
            if perception:
                error = np.array(df.loc[df.delays == 0, 'global_resid_error'])
            elif only_delay is None:
                error = np.array(df['global_resid_error'])
            else:
                error = np.array(df.loc[df.delays == only_delay,
                                        'global_resid_error'])
        else:
            error = np.array(df.loc[df.prev_delay == only_delay,
                                    'global_resid_error'])
        error_rad = np.deg2rad(error[ind])
        
        np.save(d_stim_name, d_stim_rad)
        np.save(error_name, error_rad)
    

def perform_permutation_test(data_frames, labels, concat=False,
                             use_clifford=False, future=False, task_name='',
                             two_tailed=False):

    """Compute p-values for the significance of the Gabor fit.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    labels : tuple
      Label for each data frame (subject number, an integer).

    concat : boolean (optional)
      Whether or not to concatenate the data frames into a super
      subject.

    use_clifford : boolean (optional)
      Whether or not to use Clifford's tilt model instead of the Gabor.

    future : boolean (optional)
      Whether or not to use future_d_stim as d_stim.

    task_name : string (optional)
      '' (default) or 'var_ITI'.

    two_tailed : boolean (optional)
      Whether to do two-tailed test.

    """

    results_dir = '/home/despo/dbliss/dopa_net/results/bliss_behavior/fig_1/'

    if concat:
        diff_rad = np.array([])
        error_rad = np.array([])
    
    for i, lab in enumerate(labels):

        df = data_frames[i]
        if not future:
            diff = np.array(df['d_stim'])
        else:
            diff = np.array(df['d_stim_future'])
        ind = ~np.isnan(diff)
        diff = diff[ind]

        if concat:
            diff_rad = np.concatenate([diff_rad, np.deg2rad(diff)])
        else:
            diff_rad = np.deg2rad(diff)

        error = np.array(df['global_resid_error'])

        error = error[ind]
        
        if concat:
            error_rad = np.concatenate([error_rad, np.deg2rad(error)])
        else:
            error_rad = np.deg2rad(error)

        if not concat:

            if not use_clifford:
                actual_a, actual_w, _ = fit_dog(error_rad, diff_rad)
            else:
                actual_c, actual_s, actual_m, _ = fit_clifford(error_rad,
                                                               diff_rad)

            if not use_clifford:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_dog_all_delays_' +
                        's%03d%s.txt' % (lab, task_name)))
                assert perm_res.shape[0] == 10000
                a_permuted = perm_res[:, 0]
                w_permuted = perm_res[:, 1]
                n_permutations = perm_res.shape[0]

            else:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_clifford_all_delays_' +
                        's%03d%s.txt' % (lab, task_name)))
                assert perm_res.shape[0] == 10000
                c_permuted = perm_res[:, 0]
                s_permuted = perm_res[:, 1]
                m_permuted = perm_res[:, 2]
                n_permutations = perm_res.shape[0]
                    
            if not use_clifford:

                # Compute the peak-to-peak.
                theta = np.linspace(-np.pi, np.pi, 1000)
                fit = dog(theta, actual_a, actual_w)
                p2p_actual = np.sign(actual_a) * (fit.max() - fit.min())

                # Compute the permuted peak-to-peaks.
                p2p_permuted = np.empty(n_permutations)
                for i in range(n_permutations):
                    fit = dog(theta, a_permuted[i], w_permuted[i])
                    peak_to_peak = np.sign(a_permuted[i]) * (fit.max() -
                                                             fit.min())
                    p2p_permuted[i] = peak_to_peak

                if actual_a < 0:
                    c_p = np.count_nonzero(p2p_actual >
                                           p2p_permuted) / float(
                        n_permutations)
                elif actual_a > 0:
                    c_p = np.count_nonzero(p2p_permuted >
                                           p2p_actual) / float(
                        n_permutations)
                else:
                    raise ValueError('a is zero!')
                print 'p-value:', c_p, 'p2p:', p2p_actual

            else:

                # Compute the peak-to-peak.
                theta = np.linspace(-np.pi, np.pi, 1000)
                fit = actual_m * clifford(theta, actual_c, actual_s)
                p2p_actual = actual_m * (fit.max() - fit.min())

                # Compute the permuted peak-to-peaks.
                p2p_permuted = np.empty(n_permutations)
                for i in range(n_permutations):
                    fit = m_permuted[i] * clifford(theta, c_permuted[i],
                                                   s_permuted[i])
                    peak_to_peak = m_permuted[i] * (fit.max() - fit.min())
                    p2p_permuted[i] = peak_to_peak

                if actual_m < 0:
                    c_p = np.count_nonzero(p2p_actual >
                                           p2p_permuted) / float(
                        n_permutations)
                elif actual_m > 0:
                    c_p = np.count_nonzero(p2p_permuted >
                                           p2p_actual) / float(
                        n_permutations)
                else:
                    raise ValueError('m is zero!')
                print 's%03d: p2p:' % (lab,), c_p, 'm:', actual_m
                                         
    if concat:

        actual_a, actual_w, _ = fit_dog(error_rad, diff_rad)

        sub_string = '_'.join('%03d' % (lab,) for lab in labels)
        if not future:
            perm_res = np.loadtxt(os.path.join(results_dir,
                                               'permutations_dog_all_delays' +
                                               '_s%s%s.txt'
                                               % (sub_string, task_name)))
        else:
            perm_res = np.loadtxt(os.path.join(
                    results_dir, 'permutations_dog_all_delays_future_'
                    + 's%s%s.txt' % (sub_string, task_name)))

        n_permutations = perm_res.shape[0]
        assert n_permutations == 10000

        a_permuted = perm_res[:, 0]
        w_permuted = perm_res[:, 1]

        # Compute the peak-to-peak.
        theta = np.linspace(-np.pi, np.pi, 1000)
        fit = dog(theta, actual_a, actual_w)
        p2p_actual = np.sign(actual_a) * (fit.max() - fit.min())

        # Compute the permuted peak-to-peaks.
        p2p_permuted = np.empty(n_permutations)
        for i in range(n_permutations):
            fit = dog(theta, a_permuted[i], w_permuted[i])
            peak_to_peak = np.sign(a_permuted[i]) * (fit.max() - fit.min())
            p2p_permuted[i] = peak_to_peak

        if two_tailed:
            c_p = np.count_nonzero(np.abs(p2p_permuted) >
                                   abs(p2p_actual)) / float(n_permutations)
        else:
            if np.sign(actual_a) < 0:
                c_p = np.count_nonzero(p2p_actual >
                                       p2p_permuted) / float(
                    n_permutations)
            elif np.sign(actual_a) > 0:
                c_p = np.count_nonzero(p2p_permuted >
                                       p2p_actual) / float(
                    n_permutations)
            else:
                raise ValueError('a is zero!')

        print 'p-value:', c_p, 'p2p:', np.rad2deg(p2p_actual)


def print_confidence_interval(data_frames, labels, future=False, task_name=''):
    
    results_dir = '/home/despo/dbliss/dopa_net/results/bliss_behavior/fig_1/'
    sub_string = '_'.join('%03d' % (lab,) for lab in labels)
    
    concat_diff_rad = np.array([])
    concat_error_rad = np.array([])

    for lab, df in zip(labels, data_frames):

        if not future:
            diff = np.array(df['d_stim'])
        else:
            diff = np.array(df['d_stim_future'])
        ind = ~np.isnan(diff)
        diff_rad = np.deg2rad(diff[ind])
        concat_diff_rad = np.concatenate([concat_diff_rad, diff_rad])

        error = np.array(df['global_resid_error'])
        error_rad = np.deg2rad(error[ind])
        concat_error_rad = np.concatenate([concat_error_rad, error_rad])

    theta = np.linspace(-np.pi, np.pi, 1000)
    a, w, _ = fit_dog(concat_error_rad, concat_diff_rad)
    fit = dog(theta, a, w)
    p2p = np.sign(a) * (fit.max() - fit.min())

    if not future:
        boot_res = np.loadtxt(os.path.join(results_dir,
                                           'bootstrap_dog_all_delays_s%s%s.txt'
                                           % (sub_string, task_name)))
    else:
        boot_res = np.loadtxt(os.path.join(
                results_dir, 'bootstrap_dog_all_delays_future_s%s%s.txt' %
                (sub_string, task_name)))

    n_permutations = 10000
    assert boot_res.shape[0] == n_permutations
    a_permuted = boot_res[:, 0]
    w_permuted = boot_res[:, 1]

    c_star = np.empty(n_permutations)
    for i in range(n_permutations):
        fit = dog(theta, a_permuted[i], w_permuted[i])
        c_star[i] = np.sign(a_permuted[i]) * (fit.max() - fit.min())

    delta_star = np.sort(c_star - p2p)
    delta_star_25 = delta_star[int(97.5 / 100 * n_permutations)]
    delta_star_975 = delta_star[int(2.5 / 100 * n_permutations)]
    ci_low = p2p - delta_star_25
    ci_high = p2p - delta_star_975

    print np.rad2deg(ci_low), np.rad2deg(ci_high)


def perform_permutation_test_conditions(data_frames, labels, previous=False,
                                        use_clifford=True):
    
    """Compute p-values for different delay/ITI conditions.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    labels : tuple
      Label for each data frame (subject number, an integer).

    previous : boolean (optional)
      Whether or not to use the previous trial's delay.

    """
    
    results_dir = utils._get_results_dir('fig_1', 'bliss_behavior')
    delays = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    n_delays = len(delays)
    n_permutations = 10000
    diff_rad_concat = [np.array([]) for d in delays]
    error_rad_concat = [np.array([]) for d in delays]
    for i, lab in enumerate(labels):
        df = data_frames[i]
        for j, d in enumerate(delays):
            if not previous:
                diff = np.array(df.loc[df.delays == d, 'd_stim'])
            else:
                diff = np.array(df.loc[df.prev_delay == d, 'd_stim'])
            ind = ~np.isnan(diff)
            diff_rad = np.deg2rad(diff[ind])
            if not previous:
                error = np.array(df.loc[df.delays == d, 'global_resid_error'])
            else:
                error = np.array(df.loc[df.prev_delay == d,
                                        'global_resid_error'])
            error_rad = np.deg2rad(error[ind])
            diff_rad_concat[j] = np.concatenate([diff_rad_concat[j], diff_rad])
            error_rad_concat[j] = np.concatenate([error_rad_concat[j],
                                                  error_rad])
    actual_c_values = np.empty(n_delays)
    actual_s_values = np.empty(n_delays)
    if use_clifford:
        actual_m_values = np.empty(n_delays)
    c_values = np.empty((n_permutations, n_delays))
    s_values = np.empty((n_permutations, n_delays))
    if use_clifford:
        m_values = np.empty((n_permutations, n_delays))
    sub_string = '_'.join('%03d' % (lab,) for lab in labels)
    for j, d in enumerate(delays):
        if use_clifford:
            (actual_c_values[j],
             actual_s_values[j],
             actual_m_values[j], _) = fit_clifford(error_rad_concat[j],
                                                   diff_rad_concat[j])
        else:
            (actual_c_values[j],
             actual_s_values[j], _) = fit_dog(error_rad_concat[j],
                                              diff_rad_concat[j])
        if not previous:
            if j == 0:
                if use_clifford:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_clifford_perception_s%s.txt'
                            % sub_string))
                else:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_dog_perception_s%s.txt'
                            % sub_string))
            else:
                if use_clifford:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_clifford_d%02d_s%s.txt'
                            % (d, sub_string)))
                else:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_dog_d%02d_s%s.txt'
                            % (d, sub_string)))
        else:
            if use_clifford:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_clifford_d%02d_s%s_previous.txt'
                        % (d, sub_string)))
            else:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_dog_d%02d_s%s_previous.txt'
                        % (d, sub_string)))

        assert perm_res.shape[0] == n_permutations
        c_values[:, j] = perm_res[:, 0]
        s_values[:, j] = perm_res[:, 1]
        if use_clifford:
            m_values[:, j] = perm_res[:, 2]

    # For each delay, compute the peak-to-peak.
    theta = np.linspace(-np.pi, np.pi, 1000)
    p2p_actual = np.empty(n_delays)
    for d in range(n_delays):
        if use_clifford:
            fit = actual_m_values[d] * clifford(theta, actual_c_values[d],
                                                actual_s_values[d])
            p2p_actual[d] = actual_m_values[d] * (fit.max() - fit.min())
        else:
            fit = dog(theta, actual_c_values[d], actual_s_values[d])
            p2p_actual[d] = np.sign(actual_c_values[d]) * (fit.max() -
                                                           fit.min())

    # For each delay, compute the permuted peak-to-peaks.
    p2p_permuted = np.empty((n_permutations, n_delays))
    for d in range(n_delays):
        for i in range(n_permutations):
            if use_clifford:
                fit = m_values[i, d] * clifford(theta, c_values[i, d],
                                                s_values[i, d])
                peak_to_peak = m_values[i, d] * (fit.max() - fit.min())
            else:
                fit = dog(theta, c_values[i, d], s_values[i, d])
                peak_to_peak = np.sign(c_values[i, d]) * (fit.max() -
                                                          fit.min())
            p2p_permuted[i, d] = peak_to_peak

    if not previous:
        # Print significance of each delay.
        p = np.count_nonzero(np.abs(p2p_permuted[:, 0]) >=
                             abs(p2p_actual[0])) / float(n_permutations)
        print '0:', p
        p = np.count_nonzero(p2p_permuted[:, 1] >= p2p_actual[1]) / \
                             float(n_permutations)
        print '1:', p
        p = np.count_nonzero(p2p_permuted[:, 2] >= p2p_actual[2]) / \
                             float(n_permutations)
        print '3:', p
        p = np.count_nonzero(p2p_permuted[:, 3] >= p2p_actual[3]) / \
                             float(n_permutations)
        print '6:', p
        p = np.count_nonzero(p2p_permuted[:, 4] >= p2p_actual[4]) / \
                             float(n_permutations)
        print '10:', p

        # Print significance of the comparisons.
        p = np.count_nonzero((p2p_permuted[:, 1] - p2p_permuted[:, 0]) >=
                             (p2p_actual[1] - p2p_actual[0])) / \
                             float(n_permutations)
        print '1 > 0:', p
        p = np.count_nonzero((p2p_permuted[:, 2] - p2p_permuted[:, 0]) >=
                             (p2p_actual[2] - p2p_actual[0])) / \
                             float(n_permutations)
        print '3 > 0:', p
        p = np.count_nonzero((p2p_permuted[:, 3] - p2p_permuted[:, 0]) >=
                             (p2p_actual[3] - p2p_actual[0])) / \
                             float(n_permutations)
        print '6 > 0:', p
        p = np.count_nonzero((p2p_permuted[:, 4] - p2p_permuted[:, 0]) >=
                             (p2p_actual[4] - p2p_actual[0])) / \
                             float(n_permutations)
        print '10 > 0:', p
        p = np.count_nonzero((p2p_permuted[:, 2] - p2p_permuted[:, 1]) >=
                             (p2p_actual[2] - p2p_actual[1])) / \
                             float(n_permutations)
        print '3 > 1:', p
        p = np.count_nonzero((p2p_permuted[:, 3] - p2p_permuted[:, 1]) >=
                             (p2p_actual[3] - p2p_actual[1])) / \
                             float(n_permutations)
        print '6 > 1:', p
        p = np.count_nonzero((p2p_permuted[:, 4] - p2p_permuted[:, 1]) >=
                             (p2p_actual[4] - p2p_actual[1])) / \
                             float(n_permutations)
        print '10 > 1:', p
        p = np.count_nonzero((p2p_permuted[:, 3] - p2p_permuted[:, 2]) >=
                             (p2p_actual[3] - p2p_actual[2])) / \
                             float(n_permutations)
        print '6 > 3:', p
        p = np.count_nonzero((p2p_permuted[:, 4] - p2p_permuted[:, 2]) >=
                             (p2p_actual[4] - p2p_actual[2])) / \
                             float(n_permutations)
        print '10 > 3:', p
        p = np.count_nonzero((p2p_permuted[:, 4] - p2p_permuted[:, 3]) >=
                             (p2p_actual[4] - p2p_actual[3])) / \
                             float(n_permutations)
        print '10 > 6:', p
    else:
        # Print significance of each delay.
        p = np.count_nonzero(p2p_permuted[:, 0] >=
                             p2p_actual[0]) / float(n_permutations)
        print '0:', p
        p = np.count_nonzero(p2p_permuted[:, 1] >= p2p_actual[1]) / \
                             float(n_permutations)
        print '1:', p
        p = np.count_nonzero(p2p_permuted[:, 2] >= p2p_actual[2]) / \
                             float(n_permutations)
        print '3:', p
        p = np.count_nonzero(p2p_permuted[:, 3] >= p2p_actual[3]) / \
                             float(n_permutations)
        print '6:', p
        p = np.count_nonzero(p2p_permuted[:, 4] >= p2p_actual[4]) / \
                             float(n_permutations)
        print '10:', p

        # Print significance of the comparisons.
        p = np.count_nonzero(np.abs(p2p_permuted[:, 1] -
                                    p2p_permuted[:, 0]) >=
                             abs(p2p_actual[1] - p2p_actual[0])) / \
                             float(n_permutations)
        print '1 > 0:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 2] -
                                    p2p_permuted[:, 0]) >=
                             abs(p2p_actual[2] - p2p_actual[0])) / \
                             float(n_permutations)
        print '3 > 0:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 3] -
                                    p2p_permuted[:, 0]) >=
                             abs(p2p_actual[3] - p2p_actual[0])) / \
                             float(n_permutations)
        print '6 > 0:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 4] -
                                    p2p_permuted[:, 0]) >=
                             abs(p2p_actual[4] - p2p_actual[0])) / \
                             float(n_permutations)
        print '10 > 0:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 2] -
                                    p2p_permuted[:, 1]) >=
                             abs(p2p_actual[2] - p2p_actual[1])) / \
                             float(n_permutations)
        print '3 > 1:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 3] -
                                    p2p_permuted[:, 1]) >=
                             abs(p2p_actual[3] - p2p_actual[1])) / \
                             float(n_permutations)
        print '6 > 1:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 4] -
                                    p2p_permuted[:, 1]) >=
                             abs(p2p_actual[4] - p2p_actual[1])) / \
                             float(n_permutations)
        print '10 > 1:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 3] -
                                    p2p_permuted[:, 2]) >=
                             abs(p2p_actual[3] - p2p_actual[2])) / \
                             float(n_permutations)
        print '6 > 3:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 4] -
                                    p2p_permuted[:, 2]) >=
                             abs(p2p_actual[4] - p2p_actual[2])) / \
                             float(n_permutations)
        print '10 > 3:', p
        p = np.count_nonzero(np.abs(p2p_permuted[:, 4] -
                                    p2p_permuted[:, 3]) >=
                             abs(p2p_actual[4] - p2p_actual[3])) / \
                             float(n_permutations)
        print '10 > 6:', p
    

def plot_bars(data_frames, labels, previous=False, use_clifford=True,
              ylim=None, f_name=None):

    results_dir = utils._get_results_dir('fig_1', 'bliss_behavior')
    n_subs = len(labels)
    delays = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    n_delays = len(delays)
    
    # Initialize the array for bar heights.
    delay_actual_p2p = np.empty(n_delays)

    # Initialize the arrays for concatenated diff and error.
    concat_diff_rad = [np.array([])] * n_delays
    concat_error_rad = [np.array([])] * n_delays

    # Initialize the arrays for the confidence intervals.
    ci_low = np.empty_like(delay_actual_p2p)
    ci_high = np.empty_like(delay_actual_p2p)

    for i, lab in enumerate(labels):

        df = data_frames[i]
        if not previous:
            # For each delay, make arrays that collapse over
            # subjects for d_stim and error.
            for j, d in enumerate(delays):
                diff = np.array(df.loc[df.delays == d, 'd_stim'])
                ind = ~np.isnan(diff)
                diff_rad = np.deg2rad(diff[ind])
                concat_diff_rad[j] = np.concatenate([concat_diff_rad[j],
                                                     diff_rad])
                error = np.array(df.loc[df.delays == d,
                                        'global_resid_error'])
                error_rad = np.deg2rad(error[ind])
                concat_error_rad[j] = np.concatenate([concat_error_rad[j],
                                                      error_rad])
        else:
            for j, d in enumerate(delays):
                diff = np.array(df.loc[df.prev_delay == d, 'd_stim'])
                ind = ~np.isnan(diff)
                diff_rad = np.deg2rad(diff[ind])
                concat_diff_rad[j] = np.concatenate([concat_diff_rad[j],
                                                     diff_rad])
                error = np.array(df.loc[df.prev_delay == d,
                                        'global_resid_error'])
                error_rad = np.deg2rad(error[ind])
                concat_error_rad[j] = np.concatenate([concat_error_rad[j],
                                                      error_rad])

    # For each delay, do the Clifford fit.
    theta = np.linspace(-np.pi, np.pi, 1000)
    for j in range(n_delays):
        if use_clifford:
            c, s, m, _ = fit_clifford(concat_error_rad[j], concat_diff_rad[j])
            fit = m * clifford(theta, c, s)
            delay_actual_p2p[j] = m * (fit.max() - fit.min())
        else:
            a, w, _ = fit_dog(concat_error_rad[j], concat_diff_rad[j])
            fit = dog(theta, a, w)
            delay_actual_p2p[j] = np.sign(a) * (fit.max() - fit.min())

    # Load the bootstrapping results and compute confidence intervals.
    sub_string = '_'.join('%03d' % (lab,) for lab in labels)
    n_permutations = 10000
    for j, d in enumerate(delays):
        if not previous:
            if j == 0:
                if use_clifford:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_clifford_perception_s%s.txt' %
                            sub_string))
                else:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_dog_perception_s%s.txt' %
                            sub_string))
            else:
                if use_clifford:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_clifford_d%02d_s%s.txt' %
                            (d, sub_string)))
                else:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_dog_d%02d_s%s.txt' %
                            (d, sub_string)))
        else:
            if use_clifford:
                boot_res = np.loadtxt(os.path.join(
                        results_dir,
                        'bootstrap_clifford_d%02d_previous_s%s.txt'
                        % (d, sub_string)))
            else:
                boot_res = np.loadtxt(os.path.join(
                        results_dir,
                        'bootstrap_dog_d%02d_previous_s%s.txt'
                        % (d, sub_string)))
        try:
            assert boot_res.shape[0] == n_permutations
        except AssertionError:
            raise AssertionError('wrong number bootstraps for %d' % d)
        c_permuted = boot_res[:, 0]
        s_permuted = boot_res[:, 1]
        if use_clifford:
            m_permuted = boot_res[:, 2]
        c_star = np.empty(n_permutations)
        for i in range(n_permutations):
            if use_clifford:
                fit = m_permuted[i] * clifford(theta, c_permuted[i],
                                               s_permuted[i])
                peak_to_peak = m_permuted[i] * (fit.max() - fit.min())
            else:
                fit = dog(theta, c_permuted[i], s_permuted[i])
                peak_to_peak = np.sign(c_permuted[i]) * (fit.max() - fit.min())
            c_star[i] = peak_to_peak
        # Compute the confidence interval.
        delta_star = np.sort(c_star - delay_actual_p2p[j])
        delta_star_25 = delta_star[int(97.5 / 100 * n_permutations)]
        delta_star_975 = delta_star[int(2.5 / 100 * n_permutations)]
        ci_low[j] = delay_actual_p2p[j] - delta_star_25
        ci_high[j] = delay_actual_p2p[j] - delta_star_975

    f, ax_arr = plt.subplots(1, 1, figsize=(8.2225, 5.5))
    width = 0.6

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    print np.rad2deg(delay_actual_p2p)
    ci_low = np.rad2deg(ci_low)
    ci_high = np.rad2deg(ci_high)
    delay_actual_p2p = np.rad2deg(delay_actual_p2p)
    print ci_low
    print ci_high
    plt.bar(delays - 0.4, delay_actual_p2p,
            yerr=(delay_actual_p2p - ci_low,
                  ci_high - delay_actual_p2p),
            ecolor='k', color='gray', error_kw={'linewidth': 2,
                                                'capthick': 2})

    plt.xlim(-0.9, 10.9)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.gca().set_xticks(delays)
    plt.gca().set_xticklabels(np.array(plt.gca().get_xticks(), dtype=int),
                              fontsize=18)
    plt.gca().set_yticklabels(plt.gca().get_yticks(), fontsize=18)
    if not previous:
        plt.xlabel("Length of current trial's delay period (s)", fontsize=24)
    else:
        plt.xlabel("Previous trial's delay (s)", fontsize=24)
    plt.ylabel('Serial dependence ($^\circ$)', fontsize=24)

    plt.tight_layout()
    if f_name is not None:
        plt.savefig('plot_bars_%s.png' % f_name, bbox_inches='tight')
        return
    if previous and not use_clifford:
        plt.savefig('plot_bars_previous_dog.png', bbox_inches='tight')
    if not previous:
        plt.savefig('plot_bars_wm.png', bbox_inches='tight')
    else:
        plt.savefig('plot_bars_previous.png', bbox_inches='tight')
        

def plot_running_avg_with_fit(df, f_name, bin_width=350, bin_step=10,
                              fixed_ylim=True, single=False):
    
    # Get the data.
    diff = np.array(df['d_stim'])
    ind = ~np.isnan(diff)
    diff_rad = np.deg2rad(diff[ind])
    resid_error = np.array(df['global_resid_error'])
    resid_error_rad = np.deg2rad(resid_error[ind])

    # Fit the models.
    a, w, _ = fit_dog(resid_error_rad, diff_rad)
    c, s, m, _ = fit_clifford(resid_error_rad, diff_rad)

    # Sort the data by d_stim.
    ind = diff_rad.argsort()
    diff_rad = diff_rad[ind]
    resid_error_rad = resid_error_rad[ind]

    # Pad the data.
    diff_rad = np.concatenate([diff_rad[-bin_width / 2:] - 2 * np.pi, diff_rad,
                               diff_rad[:bin_width / 2] + 2 * np.pi])
    resid_error_rad = np.concatenate([resid_error_rad[-bin_width / 2:],
                                      resid_error_rad,
                                      resid_error_rad[:bin_width / 2]])

    # Compute the running average.
    bin_starts = np.arange(0, len(diff_rad) - bin_width, bin_step)
    bin_stops = bin_starts + bin_width
    diff_means = np.empty(len(bin_starts))
    error_means = np.empty_like(diff_means)
    error_sem = np.empty_like(error_means)
    for i in range(len(bin_starts)):
        diff_means[i] = diff_rad[bin_starts[i]:bin_stops[i]].mean()
        error_selection = resid_error_rad[bin_starts[i]:bin_stops[i]]
        error_means[i] = error_selection.mean()
        error_sem[i] = error_selection.std() / np.sqrt(len(error_selection))
    
    # Convert to degrees.
    diff_means = np.rad2deg(diff_means)
    error_means = np.rad2deg(error_means)
    error_sem = np.rad2deg(error_sem)

    plt.figure(figsize=(9, 6))
    
    # Plot the data.
    plt.plot(diff_means, error_means, 'k', linewidth=1)
    plt.fill_between(diff_means,
                     error_means - error_sem,
                     error_means + error_sem,
                     alpha=0.25, color='k')

    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    theta = np.linspace(-np.pi, np.pi, 1000)
    fit = m * clifford(theta, c, s)
    p2p = m * (fit.max() - fit.min())
    print np.rad2deg(p2p)
    plt.plot(np.rad2deg(theta), np.rad2deg(m * clifford(theta, c, s)),
             'orange', linewidth=3)
    fit = dog(theta, a, w)
    p2p = np.sign(a) * (fit.max() - fit.min())
    print np.rad2deg(p2p)
    plt.plot(np.rad2deg(theta), np.rad2deg(dog(theta, a, w)), 'k',
             linewidth=3)

    if f_name == 'curr_6':
        col = 'royalblue'
        p = patches.FancyArrowPatch((0, -np.rad2deg(p2p) / 2),
                                    (0, np.rad2deg(p2p) / 2),
                                    arrowstyle='<|-|>',
                                    mutation_scale=30,
                                    edgecolor=col,
                                    facecolor=col)
        plt.gca().add_patch(p)
        plt.plot([np.rad2deg(theta[fit.argmin()]), 0],
                 [-np.rad2deg(p2p) / 2, -np.rad2deg(p2p) / 2], '--',
                 color=col, linewidth=2)
        plt.plot([0, np.rad2deg(theta[fit.argmax()])],
                 [np.rad2deg(p2p) / 2, np.rad2deg(p2p) / 2], '--',
                 color=col, linewidth=2)

    # Format the figure.
    plt.xlim(-180, 180)
    if not single:
        if f_name.startswith('prev'):
            plt.ylim(-1.5, 1.5)
        elif fixed_ylim:
            plt.ylim(-2.5, 2.5)
        else:
            plt.ylim(-2, 2)
    else:
        if fixed_ylim:
            plt.ylim(-4, 4)
    plt.gca().set_xticks([-180, -90, 0, 90, 180])
    plt.gca().set_xticklabels(np.array(plt.gca().get_xticks(), dtype=int),
                              fontsize=18)
    plt.gca().set_yticklabels(plt.gca().get_yticks(), fontsize=18)
    plt.ylabel('Error ($^\circ$)', fontsize=24)
    plt.xlabel('Relative location of previous stimulus ($^\circ$)',
               fontsize=24)

    plt.savefig('running_avg_%s.png' % f_name, bbox_inches='tight')
        

def plot_ploner_fig_3(df):

    """Plot adjusted Figure 3 from Ploner et al., '98, Eur J Neurosci.

    Parameters
    ----------
    df : pandas.DataFrame
      One data frame for all subjects.

    Notes
    -----
    Note that in Ploner et al., the dependent variable is the ratio of
    saccade amplitude to target eccentricity.  All stimuli were
    presented on the horizontal axis.  My dependent variable here is
    orthogonal to saccade amplitude; it is the error in the angle along
    the circle on which the stimuli were presented in degrees.  (No
    error for Ploner et al. results in a dependent variable value of 1;
    no error for me results in a dependent variable value of 0.)

    """

    delays = [0, 1, 3, 6, 10]

    # Compute the variance for each delay.
    variance = np.empty(len(delays))
    ci_low = np.empty_like(variance)
    ci_high = np.empty_like(variance)
    for i, d in enumerate(delays):
        # Pull out the errors.
        errors = df.loc[df.delays == d, 'errors']
        errors = errors[~np.isnan(errors)]
        # Convert to a numpy array.
        errors = np.array(errors)
        # Now compute the variance.
        variance[i] = errors.var()

        # Compute the bootstrapped variance.
        bootstrapped_variance = np.empty(10000)
        for j in range(len(bootstrapped_variance)):
            ind = np.random.choice(np.arange(len(errors)), len(errors))
            bootstrapped_variance[j] = errors[ind].var()

        # Compute the bootstrapped CIs.
        delta_star = np.sort(bootstrapped_variance - variance[i])
        delta_star_25 = delta_star[int(97.5 / 100 * 10000)]
        delta_star_975 = delta_star[int(2.5 / 100 * 10000)]
        ci_low[i] = variance[i] - delta_star_25
        ci_high[i] = variance[i] - delta_star_975

    # Fit a line to the variance.
    m, b = np.polyfit(delays, variance, 1)

    # Fit a power law to the variance.
    alpha, inner_add, beta = fit_power_law(delays, variance)
    print alpha, inner_add, beta

    # Plot the figure.
    plt.plot(delays, variance, 'k', linewidth=1)
    plt.plot(delays, m * np.array(delays) + b, 'k', linewidth=2)
    # Plot the power law.
    many_delays = np.linspace(0, 10, 1000)
    plt.plot(many_delays, alpha * (many_delays + inner_add) ** beta,
             'orange', linewidth=2)
    print ci_low
    print ci_high
    plt.fill_between(delays, ci_low, ci_high, alpha=0.25, color='k')

    # Format the figure.
    plt.xlim(-0.5, 10.5)
    plt.xlabel("Current trial's delay (s)", fontsize=24)
    plt.gca().set_xticks(delays)
    plt.gca().set_xticklabels(np.array(plt.gca().get_xticks(), dtype=int),
                              fontsize=18)
    plt.gca().set_yticklabels(plt.gca().get_yticks(), fontsize=18)
    plt.gca().set_ylabel('Variance ($^{\circ^2}$)', fontsize=24)
    plt.tight_layout()
    plt.savefig('ploner_replication.png', bbox_inches='tight')
