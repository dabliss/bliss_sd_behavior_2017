import argparse
import os
import fcntl
import sys

import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--dir', type=str, required=True)
    pa = parser.parse_args()

    np.random.seed(pa.k)

    sub_num = 39  # Do not include leading zero.
    task_name = 'exp1'

    package_dir = pa.dir

    if package_dir not in sys.path:
        sys.path.append(package_dir)
    from source import behavior_analysis as ba

    data_dir = os.path.join(package_dir, 'proc_data', task_name)

    n_permutations = 10
    params = np.empty((n_permutations, 4))

    # Load data.
    if isinstance(sub_num, int):
        f_name = os.path.join(data_dir, 's%03d' % sub_num)
        diff_rad = np.load(f_name + '_d_stim_%s_all_delays.npy' % task_name)
        resid_error_rad = np.load(f_name +
                                  '_global_resid_error_%s_all_delays.npy' %
                                  task_name)
    else:
        diff_rad = np.array([])
        resid_error_rad = np.array([])
        for n in sub_num:
            f_name = os.path.join(data_dir, 's%03d' % n)
            diff_rad = np.concatenate([diff_rad,
                                       np.load(f_name +
                                               '_d_stim_%s_all_delays.npy'
                                               % task_name)])
            resid_error_rad = np.concatenate(
                [resid_error_rad,
                 np.load(f_name + '_global_resid_error_%s_all_delays.npy'
                         % task_name)])

    # Permute d_stim.
    for i_perm in range(n_permutations):
        np.random.shuffle(diff_rad)
        params[i_perm, :] = ba.fit_clifford(resid_error_rad, diff_rad)

    # Write to the file.
    flags = os.O_CREAT | os.O_WRONLY
    results_dir = os.path.join(package_dir, 'results', task_name)
    try:
        os.makedirs(results_dir)
    except OSError:
        pass
    if isinstance(sub_num, int):
        sub_name = '%03d' % sub_num
    else:
        sub_name = '_'.join(['%03d' % (n,) for n in sub_num])
    if task_name != '':
        f_name = os.path.join(results_dir,
                              'permutations_clifford_all_delays_s%s_%s.txt' %
                              (sub_name, task_name))
    else:
        f_name = os.path.join(results_dir,
                              'permutations_clifford_all_delays_s%s.txt' %
                              (sub_name,))
    file_handle = os.open(f_name, flags)
    fcntl.flock(file_handle, fcntl.LOCK_EX)
    with os.fdopen(file_handle, 'a') as f:
        np.savetxt(f, params)
