import os

import pandas as pd
import numpy as np
from scipy import io as sio
import matlab
import matlab.engine
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import patches


MODELS = np.array(('VMRW', 'VM', 'VP', 'VMRW+attraction', 'VMRW+swap',
                   'VM_attraction', 'VP_attraction', 'VMRW_dog', 'EP_dog',
                   'VP_dog'))


def get_subject_data(subject, sessions, task, keys, indices, data_dir):

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

    data_dir : string
      Top-level directory for the project.

    Returns
    -------
    all_sessions : pandas.DataFrame
      Data for all of a subject's sessions.
    
    """

    data_dir = os.path.join(data_dir, 'data', task)

    # Load the response data for the subject.
    for session in sessions:
        results_file = os.path.join(data_dir, '%03d_%03d_results.txt' %
                                    (subject, session))
        exec 'session_%03d = pd.read_csv("%s", sep="\t")' % (session,
                                                             results_file)
    all_sessions_response = pd.concat([eval('session_%03d' % session) for
                                       session in sessions], 
                                      ignore_index=True)

    # Load the presentation data for the subject.
    for session in sessions:
        session_file = os.path.join(data_dir, 'session_details_%03d_%03d'
                                    % (subject, session))
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


def add_columns(df, first_trial_indices, last_trial_indices, task_name='exp1'):
    """Add columns to df.

    Parameters
    ----------
    df : pandas.DataFrame
      Data Frame for a subject.

    first_trial_indices : sequence
      Indices marking the first trial in every session.

    last_trial_indices : sequence
      Indices marking the last trial in every session.

    task_name : string (optional)
      Specifier for the task.

    Returns
    -------
    df : pandas.DataFrame
      Updated Data Frame.
    
    """
    key = 'stimulus_angles'
    prev_stim = np.insert(np.array(df[key])[:-1], 0, np.nan)
    future_stim = np.insert(np.array(df[key])[1:], len(df[key]) - 1,
                            np.nan)
    if task_name == 'exp1':
        prev_delay = np.insert(np.array(df.delays, dtype=float)[:-1], 0,
                               np.nan)
    # The first trial of each session (not just the overall first
    # trial) has no previous stimulus.
    prev_stim[first_trial_indices] = np.nan
    if task_name == 'exp1':
        prev_delay[first_trial_indices] = np.nan
    # The last trial of each session (not just the overall
    # last trial) has no future stimulus.
    future_stim[last_trial_indices] = np.nan
    df['prev_stim'] = prev_stim
    if task_name == 'exp1':
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


def save_for_permutation(data_frames, sub_nums, package_dir, task_name='',
                         perception=False, only_delay=None, previous=False,
                         future=False, only_iti=None):

    """Save d_stim and global_resid_error for permutation tests.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    sub_nums : tuple
      Subject numbers

    package_dir : string
      Top-level directory for the project.

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

    only_iti : integer (optional)
      Only ITI to use.

    """
    
    data_dir = os.path.join(package_dir, 'proc_data', task_name)
    try:
        os.makedirs(data_dir)
    except OSError:
        pass

    if perception:
        per_string = '_perception'
    else:
        per_string = ''

    for i, (df, sub) in enumerate(zip(data_frames, sub_nums)):
        f_name = os.path.join(data_dir, 's%03d' % sub)

        # Get d_stim_name.
        if not previous:
            try:
                d_stim_name = f_name + '_d_stim_%s%s_%02d.npy' % (task_name,
                                                                  per_string,
                                                                  only_delay)
            except TypeError:
                if only_iti:
                    d_stim_name = f_name + '_d_stim_%s_%02d.npy' % (task_name,
                                                                    only_iti)
                elif perception:
                    d_stim_name = f_name + '_d_stim_%s%s.npy' % (task_name,
                                                                 per_string)
                else:
                    if not future:
                        d_stim_name = f_name + '_d_stim_%s_all_delays.npy' % (
                            task_name,)
                    else:
                        d_stim_name = (f_name +
                                       '_d_stim_%s_all_delays_future.npy' % (
                                task_name,))
        else:
            d_stim_name = f_name + '_d_stim_%s%s_%02d_previous.npy' % (
                task_name, per_string, only_delay)

        # Get d_stim.
        if perception:
            d_stim = np.array(df.loc[df.delays == 0, 'd_stim'])
        elif only_delay is None:
            if only_iti:
                d_stim = np.array(df.loc[df.itis == only_iti, 'd_stim'])
            elif not future:
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
                error_name = (f_name + '_global_resid_error_%s%s_%02d.npy' % (
                    task_name, per_string, only_delay))
            except TypeError:
                if only_iti:
                    error_name = (f_name + '_global_resid_error_%s_%02d.npy' %
                                  (task_name, only_iti))
                elif perception:
                    error_name = (f_name + '_global_resid_error_%s%s.npy' % (
                        task_name, per_string))
                else:
                    if not future:
                        error_name = (f_name + '_global_resid_error' +
                                      '_%s_all_delays.npy' % (task_name,))
                    else:
                        error_name = (f_name + '_global_resid_error' +
                                      '_%s_all_delays_future.npy' %
                                      (task_name,))
        else:
            error_name = (f_name + '_global_resid_error_%s%s_%02d_previous.npy'
                          % (task_name, per_string, only_delay))

        # Get error.
        if not previous:
            if perception:
                error = np.array(df.loc[df.delays == 0, 'global_resid_error'])
            elif only_delay is None:
                if only_iti is None:
                    error = np.array(df['global_resid_error'])
                else:
                    error = np.array(df.loc[df.itis == only_iti,
                                            'global_resid_error'])
            else:
                error = np.array(df.loc[df.delays == only_delay,
                                        'global_resid_error'])
        else:
            error = np.array(df.loc[df.prev_delay == only_delay,
                                    'global_resid_error'])
        error_rad = np.deg2rad(error[ind])
        
        np.save(d_stim_name, d_stim_rad)
        np.save(error_name, error_rad)
    

def perform_permutation_test(data_frames, labels, package_dir, concat=False,
                             use_clifford=False, future=False, task_name='',
                             two_tailed=False):

    """Compute p-values for the significance of the Gabor fit.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    labels : tuple
      Label for each data frame (subject number, an integer).

    package_dir : string
      Top-level directory for the project.

    concat : boolean (optional)
      Whether or not to concatenate the data frames into a super
      subject.

    use_clifford : boolean (optional)
      Whether or not to use Clifford's tilt model instead of the Gabor.

    future : boolean (optional)
      Whether or not to use future_d_stim as d_stim.

    task_name : string (optional)
      Specifier for the task.

    two_tailed : boolean (optional)
      Whether to do two-tailed test.

    """

    results_dir = os.path.join(package_dir, 'results', task_name)

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
                        's%03d_%s.txt' % (lab, task_name)))
                assert perm_res.shape[0] == 10000
                a_permuted = perm_res[:, 0]
                w_permuted = perm_res[:, 1]
                n_permutations = perm_res.shape[0]

            else:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_clifford_all_delays_' +
                        's%03d_%s.txt' % (lab, task_name)))
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
                print 's%03d' % lab, 'p-value:', c_p, 'p2p:', np.rad2deg(p2p_actual)

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
                print 's%03d' % (lab,), 'p-value:', c_p, 'p2p:', np.rad2deg(p2p_actual)
                                         
    if concat:

        actual_a, actual_w, _ = fit_dog(error_rad, diff_rad)

        sub_string = '_'.join('%03d' % (lab,) for lab in labels)
        if not future:
            perm_res = np.loadtxt(os.path.join(results_dir,
                                               'permutations_dog_all_delays' +
                                               '_s%s_%s.txt'
                                               % (sub_string, task_name)))
        else:
            perm_res = np.loadtxt(os.path.join(
                    results_dir, 'permutations_dog_all_delays_future_'
                    + 's%s_%s.txt' % (sub_string, task_name)))

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


def print_confidence_interval(data_frames, labels, package_dir, future=False,
                              task_name=''):
    
    results_dir = os.path.join(package_dir, 'results', task_name)
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
                                           'bootstrap_dog_all_delays_s%s_%s.txt'
                                           % (sub_string, task_name)))
    else:
        boot_res = np.loadtxt(os.path.join(
                results_dir, 'bootstrap_dog_all_delays_future_s%s_%s.txt' %
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


def perform_permutation_test_conditions(data_frames, labels, package_dir,
                                        previous=False, use_clifford=True,
                                        task_name=''):
    
    """Compute p-values for different delay/ITI conditions.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    labels : tuple
      Label for each data frame (subject number, an integer).

    package_dir : string
      Top-level directory for the project.

    previous : boolean (optional)
      Whether or not to use the previous trial's delay.

    task_name : string (optional)
      Specifier for the task.

    """
    
    results_dir = os.path.join(package_dir, 'results', task_name)
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
                            'permutations_clifford_perception_s%s_%s.txt'
                            % (sub_string, task_name)))
                else:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_dog_perception_s%s_%s.txt'
                            % (sub_string, task_name)))
            else:
                if use_clifford:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_clifford_d%02d_s%s_%s.txt'
                            % (d, sub_string, task_name)))
                else:
                    perm_res = np.loadtxt(os.path.join(
                            results_dir,
                            'permutations_dog_d%02d_s%s_%s.txt'
                            % (d, sub_string, task_name)))
        else:
            if use_clifford:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_clifford_d%02d_s%s_previous_%s.txt'
                        % (d, sub_string, task_name)))
            else:
                perm_res = np.loadtxt(os.path.join(
                        results_dir,
                        'permutations_dog_d%02d_s%s_previous_%s.txt'
                        % (d, sub_string, task_name)))

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
    

def plot_bars(data_frames, labels, package_dir, previous=False,
              use_clifford=True, ylim=None, f_name=None, task_name=''):

    results_dir = os.path.join(package_dir, 'results', task_name)
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
                            'bootstrap_clifford_perception_s%s_%s.txt' %
                            (sub_string, task_name)))
                else:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_dog_perception_s%s_%s.txt' %
                            (sub_string, task_name)))
            else:
                if use_clifford:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_clifford_d%02d_s%s_%s.txt' %
                            (d, sub_string, task_name)))
                else:
                    boot_res = np.loadtxt(os.path.join(
                            results_dir,
                            'bootstrap_dog_d%02d_s%s_%s.txt' %
                            (d, sub_string, task_name)))
        else:
            if use_clifford:
                boot_res = np.loadtxt(os.path.join(
                        results_dir,
                        'bootstrap_clifford_d%02d_previous_s%s_%s.txt'
                        % (d, sub_string, task_name)))
            else:
                boot_res = np.loadtxt(os.path.join(
                        results_dir,
                        'bootstrap_dog_d%02d_previous_s%s_%s.txt'
                        % (d, sub_string, task_name)))
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
    plt.bar(delays, delay_actual_p2p,
            yerr=(delay_actual_p2p - ci_low,
                  ci_high - delay_actual_p2p),
            ecolor='k', color='gray', error_kw={'linewidth': 2,
                                                'capthick': 2,
                                                'capsize': 3}, align='center',
            edgecolor='k')

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


def fit_power_law(delay_values, param_values):
    """Fit a power law relating delay to a parameter.

    Parameters
    ----------
    delay_values : numpy.array
      Delay for each parameter value.

    param_values : numpy.array
      Parameter values.
    
    Returns
    -------
    float
      Additive factor for the fit.

    float
      Multiplicative factor for the fit.

    float
      Exponential factor for the fit.

    """
    def _solver(params):
        m, c, lambda_ = params
        return param_values - (m * (delay_values + c) ** lambda_)
    min_cost = np.inf
    for _ in range(100):
        params_0 = [np.random.rand() * 100 - 50,
                    np.random.rand(),
                    np.random.rand() * 4 - 2]
        try:
            result = least_squares(_solver, params_0,
                                   bounds=([-np.inf, -50, -25],
                                           [np.inf, 50, 25]))
        except ValueError:
            continue
        if result['cost'] < min_cost:
            best_params, min_cost = result['x'], result['cost']
    try:
        return best_params
    except UnboundLocalError:
        return np.nan, np.nan, np.nan

    
def plot_indiv_subs(data_frames, labels, models, package_dir, supp=False,
                    task_name=''):

    results_dir = os.path.join(package_dir, 'results', task_name)

    n_subs = len(data_frames)
    
    theta = np.linspace(-np.pi, np.pi, 1000)
    n_permutations = 10000
    p2p_values = np.empty(n_subs)
    ci_low = np.empty_like(p2p_values)
    ci_high = np.empty_like(p2p_values)
    for i, (df, lab, mod) in enumerate(zip(data_frames, labels, models)):
    
        diff = np.array(df['d_stim'])
        ind = ~np.isnan(diff)
        error = np.array(df['global_resid_error'])
        diff_rad = np.deg2rad(diff[ind])
        error_rad = np.deg2rad(error[ind])

        if mod == 'dog':
            a, w, _ = fit_dog(error_rad, diff_rad)
            fit = dog(theta, a, w)
            p2p_values[i] = np.sign(a) * (fit.max() - fit.min())
        else:
            c, s, m, _ = fit_clifford(error_rad, diff_rad)
            fit = m * clifford(theta, c, s)
            p2p_values[i] = m * (fit.max() - fit.min())

        boot_res = np.loadtxt(os.path.join(
                results_dir, 'bootstrap_%s_all_delays_s%03d_%s.txt' %
                (mod, lab, task_name)))
        assert boot_res.shape[0] == n_permutations
        
        a_boot = boot_res[:, 0]
        w_boot = boot_res[:, 1]
        if mod != 'dog':
            m_boot = boot_res[:, 2]
        p2p_star = np.empty(n_permutations)
        for j in range(n_permutations):
            if mod == 'dog':
                fit = dog(theta, a_boot[j], w_boot[j])
                p2p_star[j] = np.sign(a_boot[j]) * (fit.max() - fit.min())
            else:
                fit = m_boot[j] * clifford(theta, a_boot[j], w_boot[j])
                p2p_star[j] = m_boot[j] * (fit.max() - fit.min())

        delta_star = np.sort(p2p_star - p2p_values[i])
        delta_star_25 = delta_star[int(97.5 / 100 * n_permutations)]
        delta_star_975 = delta_star[int(2.5 / 100 * n_permutations)]
        ci_low[i] = p2p_values[i] - delta_star_25
        ci_high[i] = p2p_values[i] - delta_star_975

    p2p_values = np.rad2deg(p2p_values)
    ci_low = np.rad2deg(ci_low)
    ci_high = np.rad2deg(ci_high)

    print p2p_values

    subs = np.arange(n_subs) * 5
    models = np.array(models)
    dog_ind = np.where(models == 'dog')[0]
    dog_subs = subs[dog_ind]
    dog_p2p = p2p_values[dog_ind]
    dog_ci_low = ci_low[dog_ind]
    dog_ci_high = ci_high[dog_ind]
    cliff_ind = np.where(models == 'clifford')[0]
    cliff_subs = subs[cliff_ind]
    cliff_p2p = p2p_values[cliff_ind]
    cliff_ci_low = ci_low[cliff_ind]
    cliff_ci_high = ci_high[cliff_ind]

    sig_dog_adapt_ind = np.array([0, 1, 2, 3, 4, 5, 6])
    sig_dog_sd_ind = np.array([17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34])

    plt.figure(figsize=(9, 6))
    plt.errorbar(dog_subs, dog_p2p, yerr=(dog_p2p - dog_ci_low,
                                          dog_ci_high - dog_p2p),
                 fmt='o', color='k', ecolor='k', elinewidth=2,
                 markeredgewidth=2)
    plt.errorbar(cliff_subs, cliff_p2p, yerr=(cliff_p2p - cliff_ci_low,
                                              cliff_ci_high - cliff_p2p),
                 fmt='o', color='orange', ecolor='orange', elinewidth=2,
                 markeredgewidth=2, markeredgecolor='orange')
    if not supp:
        plt.text(0.5, dog_ci_high[0] + 0.5, 'b', ha='center', fontsize=24)
        plt.text(17 * 5 + 0.1, cliff_ci_high[1] + 1.2, 'c', ha='center',
                 fontsize=24)
        plt.text(35 * 5 + 0.1, dog_ci_high[-3] + 1.2, 'd', ha='center',
                 fontsize=24)
    else:
        # Use index * 5 + whatever is needed to get it centered.
        plt.text(0.5, dog_ci_high[0] + 0.5, 's', ha='center', fontsize=18)
        plt.text(7 * 5 + 0.1, dog_ci_high[7] + 0.5, 's', ha='center',
                 fontsize=18)
        # Clifford indices are 16, 17,  21.
        plt.text(19 * 5 + 0.1, dog_ci_high[17] + 1.2, 's', ha='center',
                 fontsize=18)
        plt.text(28 * 5 + 0.1, dog_ci_high[25] + 1.2, 's', ha='center',
                 fontsize=18)
        plt.text(29 * 5 + 0.1, dog_ci_high[26] + 1.2, 's', ha='center',
                 fontsize=18)
        plt.text(30 * 5 + 0.1, dog_ci_high[27] + 1.2, 's', ha='center',
                 fontsize=18)
        plt.text(31 * 5 + 0.1, dog_ci_high[28] + 1.2, 's', ha='center',
                 fontsize=18)
        plt.text(36 * 5 + 0.1, dog_ci_high[33] + 1.2, 's', ha='center',
                 fontsize=18)
    for i in sig_dog_adapt_ind:
        plt.text(i * 5, dog_ci_low[i] - 1.0, '*', ha='center', fontsize=24)
    for i in sig_dog_sd_ind:
        plt.text(dog_subs[i] - 0.1, dog_ci_high[i] - 0.1, '*', ha='center',
                 fontsize=24)
    for i in range(len(cliff_subs)):
        plt.text(cliff_subs[i] - 0.1, cliff_ci_high[i] - 0.1, '*', ha='center',
                 fontsize=24)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlim(-5, n_subs * 5)
    plt.gca().set_yticklabels(plt.gca().get_yticks(), fontsize=18)
    plt.gca().set_xticklabels([])
    plt.xlabel('Individual participants', fontsize=24)
    plt.ylabel('Serial dependence ($^\circ$)', fontsize=24)

    plt.tight_layout()
    if not supp:
        plt.savefig('indiv_subs.png', bbox_inches='tight')
    else:
        plt.savefig('indiv_subs_supp.png', bbox_inches='tight')
    

def save_for_matlab(data_frames, labels, package_dir, task_name):
    """Save the data for model fitting in Matlab.

    Parameters
    ----------
    data_frames : tuple
      One data frame for each subject.

    labels : tuple
      Label for each data frame (subject number, an integer).

    package_dir : string
      Top-level directory for the project.

    task_name : string
      Specifier for the task.
      
    """
    data_dir = os.path.join(package_dir, 'proc_data', task_name)
    for lab, df in zip(labels, data_frames):
        for d in sorted(df.delays.unique()):
            mat_name = 's%03d_%02d' % (lab, d)
            d_stim_name = mat_name + '_d_stim'
            data = np.deg2rad(np.array(df.loc[df.delays == d, 'errors']))
            d_stim = np.deg2rad(np.array(df.loc[df.delays == d, 'd_stim']))
            # Set d_stim for first trial from each session to 0.
            d_stim[np.isnan(d_stim)] = 0
            sio.savemat(os.path.join(data_dir, '%s.mat' % mat_name),
                        {mat_name: data})
            sio.savemat(os.path.join(data_dir, '%s.mat' % d_stim_name),
                        {d_stim_name: d_stim})
        

def load_all_fits(labels, package_dir, task_name):
    """Load fits for all models.

    Returns
    -------
    all_fits : list
      All the fits.

    """
    models = ('bays_no_mu', 'EPA_no_mu', 'VPA_no_mu', 'VMRW_attraction',
              'VMRW_swap', 'EP_H_NM', 'VP_H_NM', 'VMRW_dog', 'EP_dog_NM',
              'VP_dog_NM')
    all_fits = []
    for mod in models:
        all_fits.append(load_fits(labels, mod, package_dir, task_name))
    return all_fits
            

def load_fits(subs, model, package_dir, task_name):
    """Load fits for a model.

    Parameters
    ----------
    subs : sequence
      Subject labels.

    model : string
      'bays', 'bays_with_guessing', 'bays_with_history', 'bays_GH',
      'EPA', 'EP_with_guessing', 'EP_with_history', 'EP_GH', 'VPA',
      'VP_with_guessing', 'VP_with_history', or 'VP_GH'.

    package_dir : string
      Top-level directory for the project.

    task_name : string
      Specifier for the task.

    Returns
    -------
    fit_dict : dictionary
      Dictionary with fits.

    """
    data_dir = os.path.join(package_dir, 'results', task_name)
    delays = [0, 1, 3, 6, 10]
    fit_dict = {}
    # Determine whether this model has any fixed parameters.
    no_fixed = (model.endswith('no_mu') or model.endswith('swap') or
                model == 'VMRW_attraction' or 'H_NM' in model or
                'dog' in model)
    for lab in subs:
        for d in delays:
            if no_fixed:
                key = '%03d_%02d' % (lab, d)
            else:
                key = '%03d' % (lab,)
            fit_dict[key] = {}
            attempt = 1
            while True:
                if no_fixed:
                    f_name = 's%03d_%02d_%s_%02d' % (lab, d, model, attempt)
                else:
                    f_name = 's%03d_%s_%02d' % (lab, model, attempt)
                try:
                    fit = sio.loadmat(os.path.join(data_dir, f_name))
                except IOError:
                    break
                except TypeError:
                    raise TypeError('%s is fucked up.' % f_name)
                except:
                    raise TypeError('%s is fucked up.' % f_name)
                else:
                    fit_dict[key][attempt] = fit
                    attempt += 1
            # Remove all but the best attempt here.  Note this assumes
            # there is at least one attempt.
            reduced_dict = {}
            try:
                reduced_dict[1] = fit_dict[key][get_best_attempt(fit_dict[key])]
            except UnboundLocalError:
                raise TypeError('No attempts for %s %s' % (key, model))
            fit_dict[key] = reduced_dict
            if not no_fixed:
                break
    return fit_dict


def get_best_attempt(model, attempt_range=None):
    """Return number of best attempt for a model.

    Parameters
    ----------
    model : dictionary
      Fits for a particular model and dataset.

    attempt_range : sequence
      Lowest and highest attempt numbers to consider.

    Returns
    -------
    best_attempt : integer
      Number of the best attempt.

    """
    if attempt_range is None:
        attempt_low = 1
        attempt_high = len(model.keys())
    else:
        attempt_low, attempt_high = attempt_range
    best_aic = np.inf
    for attempt in range(attempt_low, attempt_high + 1):
        try:
            aic = model[attempt]['aic'][0][0]
        except KeyError:
            aic = -model[attempt]['log_like'][0][0]
        if aic < best_aic:
            best_aic = aic
            best_attempt = attempt
    return best_attempt


def plot_model_fits(fits, labels, which_models=None, exclude_models=None,
                    specifier=''):

    """Plot average AICc for each model.

    Parameters
    ----------
    fits : tuple
      Tuple of fit dictionaries.

    which_models : sequence (optional)
      Which models to plot.

    exclude_models : sequence (optional)
      Which models not to plot.

    specifier : string (optional)
      Tack-on for the file name.

    """

    # Set n_models, fits, and models based on which subset is desired.
    models = copy.deepcopy(MODELS)
    if exclude_models is not None:
        n_models = len(models) - len(exclude_models)
        fits = copy.deepcopy(fits)
        for mod in exclude_models:
            i = np.where(models == mod)[0][0]
            fits.pop(i)
            models = np.delete(models, i)
    elif which_models is not None:
        n_models = len(which_models)
        sub_fits = []
        sub_models = []
        for mod in which_models:
            i = np.where(models == mod)[0][0]
            sub_fits.append(fits[i])
            sub_models.append(models[i])
        fits = sub_fits
        models = np.array(sub_models)
    else:
        n_models = len(fits)

    # For each subject and model, compute an AICc value.
    n_subs = len(labels)
    aic_c_values = np.empty((n_models, n_subs))
    total_n = 1000.0  # Ugly hard-coding.
    for j, model in enumerate(fits):
        # Determine whether fit is hierarchical.
        datasets = sorted(model)
        try:
            s, d = datasets[0].split('_')
        except ValueError:
            hierarchical_fit = True
        else:
            hierarchical_fit = False
        # Get log likelihood for each subject.
        log_likes = np.zeros(n_subs)
        for dset in datasets:
            try:
                best_attempt = get_best_attempt(model[dset])
            except UnboundLocalError:
                # There are no attempts for this dataset.
                raise ValueError('%s, %s has no attempts' % (models[j], dset))
            fit_details = model[dset][best_attempt]
            if hierarchical_fit:
                i_sub = labels.index(int(dset))
                log_likes[i_sub] += fit_details['log_like'][0][0]
                total_k = len(fit_details['params_hat'][0])
            else:
                aic = fit_details['aic'][0][0]
                k = len(fit_details['params_hat'][0])
                s = dset.split('_')[0]
                i_sub = labels.index(int(s))
                log_likes[i_sub] += k - aic / 2
                total_k = k * 5

        # Compute the AICc from the log likelihood for each subject.
        aic_values = 2 * total_k - 2 * log_likes
        correction = 2 * total_k * (total_k + 1) / (total_n - total_k - 1)
        aic_c_values[j, :] = aic_values + correction

    # Redefine aic_c_values to be relative to value for VM.
#    try:
#        aic_c_values -= aic_c_values[np.where(models == 'VM')[0][0]]
#    except IndexError:
#        try:
#            aic_c_values -= aic_c_values[np.where(models == 'VMRW')[0][0]]
#        except IndexError:
#            aic_c_values -= aic_c_values[np.where(models == 'VM_attraction')[0][0]]

    # Put the means in descending order.
#    ind = aic_c_values.argsort()[::-1]

    # Plot the figure.
#    lefts = np.arange(n_models, dtype='float')
#    width = 1.0
    if exclude_models is not None:
        print models[0], '-', models[2]
        print (aic_c_values[0, :] - aic_c_values[2, :]).mean()
        print (aic_c_values[0, :] - aic_c_values[2, :]).std() / np.sqrt(n_subs)
        print
        print models[2], '-', models[1]
        print (aic_c_values[2, :] - aic_c_values[1, :]).mean()
        print (aic_c_values[2, :] - aic_c_values[1, :]).std() / np.sqrt(n_subs)
        print
        print models[0], '-', models[1]
        print (aic_c_values[0, :] - aic_c_values[1, :]).mean()
        print (aic_c_values[0, :] - aic_c_values[1, :]).std() / np.sqrt(n_subs)
    elif which_models[0] == 'VMRW':
        print models[1], '-', models[0]
        print (aic_c_values[1, :] - aic_c_values[0, :]).mean()
        print (aic_c_values[1, :] - aic_c_values[0, :]).std() / np.sqrt(n_subs)
        print
        print models[2], '-', models[0]
        print (aic_c_values[2, :] - aic_c_values[0, :]).mean()
        print (aic_c_values[2, :] - aic_c_values[0, :]).std() / np.sqrt(n_subs)
    else:
        print models[1], '-', models[2]
        print (aic_c_values[1, :] - aic_c_values[2, :]).mean()
        print (aic_c_values[1, :] - aic_c_values[2, :]).std() / np.sqrt(n_subs)
        print
        print models[2], '-', models[0]
        print (aic_c_values[2, :] - aic_c_values[0, :]).mean()
        print (aic_c_values[2, :] - aic_c_values[0, :]).std() / np.sqrt(n_subs)
#    print aic_c_values[ind]
#    barlist = plt.barh(lefts, aic_c_values[ind], width, ecolor='k',
#                       color='gray')
#    plt.yticks(lefts + width / 2)
#    plt.gca().set_yticklabels([m.replace('_', '+') for m in models[ind]],
#                              fontsize=18)
#    if which_models is None:
#        plt.xlabel('$\Delta$AICc from VM', fontsize=24)
#    elif 'VP_attraction' not in which_models:
#        plt.xlabel('$\Delta$AICc from VMRW', fontsize=24)
#    else:
#        plt.xlabel('$\Delta$AICc from VM+attraction', fontsize=24)
#    plt.gca().set_xticklabels(np.array(plt.gca().get_xticks(), dtype=int),
#                              fontsize=18)
#    y_lim = plt.ylim()
#    plt.ylim(y_lim[0] - 0.25, y_lim[1] + 0.25)
#    plt.axvline(0, color='k', linewidth=1)
#    plt.tight_layout()
#    plt.savefig('plot_model_fits%s.png' % (specifier,), bbox_inches='tight')
