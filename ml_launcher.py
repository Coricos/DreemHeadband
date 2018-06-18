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
    prs.add_argument('-m', '--marker', help='Giving identity to the model', type=str, default=None)
    prs.add_argument('-x', '--max_iter', help='Number of hyperband iterations', type=int, default=100)
    prs.add_argument('-n', '--name', help='Type of model to use', type=str, default='XGB')
    prs.add_argument('-t', '--threads', help='Amount of threads', type=int, default=multiprocessing.cpu_count())
    # Parse the arguments
    prs = prs.parse_args()

    # Launch the model
    mod = ML_Model('./dataset/DTB_Headband.h5', threads=prs.threads)
    mod.learn(prs.name, marker=prs.marker, max_iter=prs.max_iter)