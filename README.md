# bliss_sd_behavior_2017

Code needed to reproduce the analysis of behavioral data in Bliss et al., 2017.

Please send bug reports and questions to dpb6@nyu.edu.

## Recipe

(1) Open notebooks/experiment_one.ipynb.

(2) Execute every cell in the notebook in order, stopping only at labeled STOP
    POINTS.

    Stop Point 1: Run scripts/perform_permutation_test_dog_all_delays.py
                  1,000 times (--k 1 to --k 1000).
