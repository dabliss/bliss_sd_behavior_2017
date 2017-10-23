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

import dopa_net.behavioral_experiments.analysis_code.behavior_analysis as ba
from dopa_net import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int)
    pa = parser.parse_args()

    np.random.seed(pa.k)

    sub_num = (1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 18, 20, 21, 24, 25, 26, 27, 28,
               29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47,
               48, 49, 50, 56)  # Do not include leading zero.
    task_name = ''

    package_dir = '/home/despo/dbliss/dopa_net/'
    data_dir = package_dir + 'behavioral_experiments/psychtoolbox/data/'

    n_permutations = 10
    params = np.empty((n_permutations, 3))

    # Load data.
    if isinstance(sub_num, int):
        f_name = data_dir + 's%03d' % sub_num
        diff_rad = np.load(f_name + '_d_stim%s_all_delays_future.npy' %
                           task_name)
        resid_error_rad = np.load(f_name +
                                  '_global_resid_error%s_all_delays_future.npy'
                                  % task_name)
    else:
        diff_rad = np.array([])
        resid_error_rad = np.array([])
        for n in sub_num:
            f_name = data_dir + 's%03d' % n
            diff_rad = np.concatenate(
                [diff_rad,
                 np.load(f_name + '_d_stim%s_all_delays_future.npy' %
                         task_name)])
            resid_error_rad = np.concatenate(
                [resid_error_rad,
                 np.load(f_name +
                         '_global_resid_error%s_all_delays_future.npy' %
                         task_name)])

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
    if task_name != '':
        f_name = os.path.join(results_dir,
                              'bootstrap_dog_all_delays_future_s%s_%s.txt' %
                              (sub_name, task_name))
    else:
        f_name = os.path.join(results_dir,
                              'bootstrap_dog_all_delays_future_s%s.txt' %
                              (sub_name,))
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
