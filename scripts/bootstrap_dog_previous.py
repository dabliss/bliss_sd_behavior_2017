"""Perform permutation test for a single subject.

Before running this,

(1) set sub_num below

(2) delete perm_k.txt (if one exists)

(3) run calls in plot_vswm_task_results.ipynb up to
    perform_permutation_test for this subject

"""


import argparse
import os
import errno
import fcntl

import numpy as np
import matplotlib.pyplot as plt

import dopa_net.behavioral_experiments.analysis_code.behavior_analysis as ba
from dopa_net import utils


def plot_avg(diff_rad, resid_error_rad):

    # Fit the Clifford model.
    a, w, m_dog, _ = ba.fit_dog(resid_error_rad, diff_rad)
    c, s, m, _ = ba.fit_clifford(resid_error_rad, diff_rad)

    bin_width = 700
    bin_step = 20
    
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
    plt.plot(np.rad2deg(theta), np.rad2deg(m * ba.clifford(theta, c, s)),
             'orange', linewidth=3)
    fit = m_dog * ba.dog(theta, a, w)
    p2p = m_dog * (fit.max() - fit.min())
    plt.plot(np.rad2deg(theta), np.rad2deg(m_dog * ba.dog(theta, a, w)), 'k',
             linewidth=3)

    # Format the figure.
    plt.xlim(-180, 180)
    plt.gca().set_xticks([-180, -90, 0, 90, 180])
    plt.gca().set_xticklabels(np.array(plt.gca().get_xticks(), dtype=int),
                              fontsize=18)
    plt.gca().set_yticklabels(plt.gca().get_yticks(), fontsize=18)
    plt.ylabel('Error ($^\circ$)', fontsize=24)
    plt.xlabel('Relative location of previous stimulus ($^\circ$)',
               fontsize=24)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int)
    pa = parser.parse_args()

    np.random.seed(pa.k)

    sub_num = (1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 18, 20, 21, 24, 25, 26, 27, 28,
               29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47,
               48, 49, 50, 56)  # Do not include leading zero.
    only_delay = 0

    package_dir = '/home/despo/dbliss/dopa_net/'
    data_dir = package_dir + 'behavioral_experiments/psychtoolbox/data/'

    n_permutations = 10
    params = np.empty((n_permutations, 4))

    # Load data.
    if isinstance(sub_num, int):
        f_name = data_dir + 's%03d' % sub_num
        diff_rad = np.load(f_name + '_d_stim%02d_previous.npy' % only_delay)
        resid_error_rad = np.load(f_name +
                                  '_global_resid_error%02d_previous.npy' %
                                  only_delay)
    else:
        diff_rad = np.array([])
        resid_error_rad = np.array([])
        for n in sub_num:
            f_name = data_dir + 's%03d' % n
            diff_rad = np.concatenate([diff_rad,
                                       np.load(f_name +
                                               '_d_stim%02d_previous.npy' %
                                               only_delay)])
            resid_error_rad = np.concatenate(
                [resid_error_rad,
                 np.load(f_name + '_global_resid_error%02d_previous.npy' %
                         only_delay)])

    n_datapoints = len(diff_rad)

    # Permute d_stim.
    for i_perm in range(n_permutations):
        ind = np.random.choice(np.arange(n_datapoints), n_datapoints)
        params[i_perm, :] = ba.fit_dog(resid_error_rad[ind], diff_rad[ind])

    # Write to the file.
    flags = os.O_CREAT | os.O_WRONLY
    results_dir = utils._get_results_dir('fig_1', 'bliss_behavior')
    utils._make_results_dir(results_dir)
    if isinstance(sub_num, int):
        sub_name = '%03d' % sub_num
    else:
        sub_name = '_'.join(['%03d' % (n,) for n in sub_num])
    f_name = os.path.join(results_dir,
                          'bootstrap_dog_d%02d_previous_s%s.txt' %
                          (only_delay, sub_name))
    file_handle = os.open(f_name, flags)
    fcntl.flock(file_handle, fcntl.LOCK_EX)
    with os.fdopen(file_handle, 'a') as f:
        np.savetxt(f, params)

    # Register that this job succeeded.
    f_name = os.path.join(results_dir, 'perm_k.txt')
    file_handle = os.open(f_name, flags)
    fcntl.flock(file_handle, fcntl.LOCK_EX)
    with os.fdopen(file_handle, 'a') as f:
        f.write('%d\n' % pa.k)
