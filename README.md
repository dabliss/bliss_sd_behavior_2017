# bliss_sd_behavior_2017

Code needed to reproduce the analysis of behavioral data in Bliss et al., 2017.

Please send bug reports and questions to dpb6@nyu.edu.

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

    Stop Point 7: In scripts/bootstrap_dog_perception.py, change sub_name to 8.
                  Then run this script 1,000 times.

                  In scripts/bootstrap_dog_wm.py, change sub_name to 8.  Change
                  only_delay to 1.  Then run it 1,000 times.

                  Run scripts/bootstrap_dog_wm.py for the other delays (3, 6,
                  and 10).

                  Then repeat all the steps in this stop point for sub_names
                  49, 26, and 1.

    Stop Point 8: In scripts/perform_permutation_test_dog_all_delays.py, change
                  sub_name to 8.  Then run this script 1,000 times.

                  Repeat this for each of the other sub_names in the cell
                  immediately below STOP POINT 8 in the notebook.

                  Repeat this process for scripts/bootstrap_dog_all.py.

    Stop Point 9: Run scripts/perform_permutation_test_clifford_all_delays.py
                  1,000 times.

                  Repeat this for each of the other sub_names in the cell
                  immediately below STOP POINT 9 in the notebook.

                  Repeat this process for scripts/bootstrap_clifford_all.py.