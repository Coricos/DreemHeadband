# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse, warnings

from package.ml_model import *
from package.dl_model import *

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-o', '--objectif', help='Objectif of cross-validation', type=str, default='dl')
    prs.add_argument('-n', '--name', help='Type of model to use', type=str, default='XGB')
    prs.add_argument('-f', '--folds', help='Number of folds for cross-validation', type=int, default=7)
    prs.add_argument('-t', '--threads', help='Amount of threads', type=int, default=multiprocessing.cpu_count())
    prs.add_argument('-l', '--log_file', help='Where to write out the intermediate scores', type=str, default='./models/CV_SCORING.log')
    # Parse the arguments
    prs = prs.parse_args()

    # Launch cross-validation for ml models
    if prs.objectif == 'ml':
        mod = CV_ML_Model('./dataset/sca_train.h5', k_fold=prs.folds, threads=prs.threads)
        mod.launch(prs.name, log_file=prs.log_file)

    # Launch cross-validation for dl models
    if prs.objectif == 'dl':
        mod = CV_DL_Model(storage='./dataset', n_iter=prs.folds)
        mod.launch(log_file=prs.log_file)


