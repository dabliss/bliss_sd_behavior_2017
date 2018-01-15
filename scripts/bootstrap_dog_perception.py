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

    sub_num = (1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 18, 20, 21, 24, 25, 26, 27, 28,
               29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47,
               48, 49, 50, 56)  # Do not include leading zero.
    task_name = 'exp1'

    package_dir = pa.dir

    if package_dir not in sys.path:
        sys.path.append(package_dir)
    from source import behavior_analysis as ba

    data_dir = os.path.join(package_dir, 'proc_data', task_name)

    n_permutations = 10
    params = np.empty((n_permutations, 3))

    # Load data.
    if isinstance(sub_num, int):
        f_name = os.path.join(data_dir, 's%03d' % sub_num)
        diff_rad = np.load(f_name + '_d_stim_%s_perception.npy' % task_name)
        resid_error_rad = np.load(f_name +
                                  '_global_resid_error_%s_perception.npy' %
                                  task_name)
    else:
        diff_rad = np.array([])
        resid_error_rad = np.array([])
        for n in sub_num:
            f_name = os.path.join(data_dir, 's%03d' % n)
            diff_rad = np.concatenate([diff_rad,
                                       np.load(f_name +
                                               '_d_stim_%s_perception.npy' %
                                               task_name)])
            resid_error_rad = np.concatenate(
                [resid_error_rad,
                 np.load(f_name + '_global_resid_error_%s_perception.npy' %
                         task_name)])

    n_datapoints = len(diff_rad)

    # Permute d_stim.
    for i_perm in range(n_permutations):
        ind = np.random.choice(np.arange(n_datapoints), n_datapoints)
        params[i_perm, :] = ba.fit_dog(resid_error_rad[ind], diff_rad[ind])

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
    f_name = os.path.join(results_dir,
                          'bootstrap_dog_perception_s%s_%s.txt' %
                          (sub_name, task_name))
    file_handle = os.open(f_name, flags)
    fcntl.flock(file_handle, fcntl.LOCK_EX)
    with os.fdopen(file_handle, 'a') as f:
        np.savetxt(f, params)
