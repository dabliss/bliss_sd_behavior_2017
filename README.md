# bliss_sd_behavior_2017

Code needed to reproduce the analysis of behavioral data in Bliss et al., 2017, 
Sci Rep.

For access to the data, please contact Dan Bliss at dpb6@nyu.edu.

This package depends on code Paul Bays has written and made available at
www.paulbays.com/code/JN14.  Paul's code should be unzipped in the matlab
directory of this repo.

Please send bug reports and any questions to dpb6@nyu.edu as well.

## Recipe

(1) Open notebooks/experiment_one.ipynb.

(2) Execute every cell in the notebook in order, stopping only at labeled STOP
    POINTS.

    Stop Point 1: Run scripts/perform_permutation_test_dog_all_delays.py
                  1,000 times (--k 1 to --k 1000).

    Stop Point 2: Run scripts/bootstrap_dog_all.py 1,000 times (--k 1 to --k
                  1000).

    Stop Point 3: Run scripts/perform_permutation_test_dog_all_delays_future.py
                  1,000 times (--k 1 to --k 1000).

    Stop Point 4: Run scripts/bootstrap_dog_all_future.py 1,000 times (--k 1 to
                  --k 1000).

    Stop Point 5: Run scripts/perform_permutation_test_dog_perception.py 1,000
                  times (--k 1 to --k 1000).

                  Then run scripts/perform_permutation_test_dog_wm.py 1,000
                  times (--k 1 to --k 1000).

                  Update the value of only_delay in
                  perform_permutation_test_dog_wm.py to be 3.  Then rerun it
                  1,000 times.

                  Repeat this process for the other delays (6 and 10).

    Stop Point 6: Run scripts/bootstrap_dog_perception.py 1,000 times (--k 1 to
                  --k 1000).

                  Then run scripts/bootstrap_dog_wm.py 1,000 times (--k 1 to
                  --k 1000).

                  Update the value of only_delay in bootstrap_dog_wm.py to be
                  3.  Then rerun it 1,000 times.

                  Repeat this process for the other delays (6 and 10).

    Stop Point 7: In scripts/bootstrap_dog_perception.py, change sub_num to 8.
                  Then run this script 1,000 times.

                  In scripts/bootstrap_dog_wm.py, change sub_num to 8.  Change
                  only_delay to 1.  Then run it 1,000 times.

                  Run scripts/bootstrap_dog_wm.py for the other delays (3, 6,
                  and 10).

                  Then repeat all the steps in this stop point for sub_num
                  49, 26, and 1.

    Stop Point 8: In scripts/perform_permutation_test_dog_all_delays.py, change
                  sub_num to 8.  Then run this script 1,000 times.

                  Repeat this for each of the other sub_nums in the cell
                  immediately below STOP POINT 8 in the notebook.

                  Repeat this process for scripts/bootstrap_dog_all.py.

    Stop Point 9: Run scripts/perform_permutation_test_clifford_all_delays.py
                  1,000 times.

                  Repeat this for each of the other sub_nums in the cell
                  immediately below STOP POINT 9 in the notebook.

                  Repeat this process for scripts/bootstrap_clifford_all.py.

    Stop Point 10: Run scripts/perform_permutation_test_dog_past_delay.py 1,000
                   times (--k 1 to --k 1000).

                   Update the value of only_delay in
                   perform_permutation_test_dog_past_delay.py to be 1.  Then
                   rerun it 1,000 times.

                   Repeat this process for the other delays (3, 6, and 10).

                   Run scripts/perform_permutation_test_clifford_past_delay.py
                   1,000 times (--k 1 to --k 1000).

                   Update the value of only_delay in
                   perform_permutation_test_clifford_past_delay.py to be 1.
                   Then rerun it 1,000 times.

                   Repeat this process for the other delays (3, 6, and 10).

    Stop Point 11: Run scripts/bootstrap_dog_previous.py 1,000 times (--k 1 to
                   --k 1000).

                   Update the value of only_delay in bootstrap_dog_previous.py
                   to be 1.  Then rerun it 1,000 times.

                   Repeat this process for the other delays (3, 6, and 10).

                   Run scripts/bootstrap_clifford_previous.py 1,000 times (--k
                   1 to --k 1000).

                   Update the value of only_delay in
                   bootstrap_clifford_previous.py to be 1.  Then rerun it 1,000
                   times.

                   Repeat this process for the other delays (3, 6, and 10).

    Stop Point 12: Run the following Matlab scripts (in the matlab folder) for
                   all subjects and delays:

                   save_EPA_fits_no_mu.m
                   save_VPA_fits_no_mu.m
                   save_bays_fits_no_mu.m
                   save_VMRW_dog_fits.m
                   save_VMRW_swap_fits.m
                   save_EP_dog_NM_fits.m
                   save_VP_dog_NM_fits.m

                   Each of these scripts takes two arguments: the full path to
                   the top-level directory of the repository and an integer k 
                   that uniquely specifies a subject/delay combination.  This k
                   value is computed as follows:

                   k = (n - 1) * 5 + d,

                   where n is the subject number and d is the delay number
                   between 1 and 5 (0 s = 1, 1 s = 2, 3 s = 3, 6 s = 4, 10 s =
                   5).

(3) Open notebooks/experiment_two.ipynb.

(4) Execute every cell in the notebook in order, stopping only at labeled STOP
    POINTS.

    Stop Point 1: In scripts/perform_permutation_test_dog_all_delays.py,
                  replace sub_num with a tuple of the sub_nums for Experiment
                  2.  Set task_name to 'exp2'.  Then run this script 1,000 
                  times (--k 1 to --k 1000).

    Stop Point 2: In scripts/bootstrap_dog_all.py, replace sub_num with a tuple
                  of the sub_nums for Experiment 2.  Set task_name to 'exp2'.  
                  Then run this script 1,000 times (--k 1 to --k 1000).

    Stop Point 3: In scripts/perform_permutation_test_dog_all_delays_future.py,
                  replace sub_num with a tuple of the sub_nums for Experiment
                  2.  Set task_name to 'exp2'.  Then run this script 1,000 
                  times (--k 1 to --k 1000).

    Stop Point 4: In scripts/bootstrap_dog_all_future.py, replace sub_num with 
                  a tuple of the sub_nums for Experiment 2.  Set task_name to
                  'exp2'.  Then run this script 1,000 times (--k 1 to --k 
                  1000).

