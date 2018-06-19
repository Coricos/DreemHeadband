# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse, warnings

from package.ml_model import *

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-n', '--name', help='Type of model to use', type=str, default='XGB')
    prs.add_argument('-f', '--folds', help='Number of folds for cross-validation', type=str, default=7)
    prs.add_argument('-t', '--threads', help='Amount of threads', type=int, default=multiprocessing.cpu_count())
    prs.add_argument('-l', '--log_file', help='Where to write out the intermediate scores', type=str, default='./nohup/CV_SCORING.log')
    # Parse the arguments
    prs = prs.parse_args()

    # Launch the model
    mod = CV_Model('./dataset/sca_train.h5', k_fold=prs.folds, threads=prs.threads)
    mod.launch(prs.name, log_file=prs.log_file)