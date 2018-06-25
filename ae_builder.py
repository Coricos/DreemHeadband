# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse

from package.ds_model import *

# Launch the pipeline
if __name__ == '__main__':

    # Initialize the arguments
    storage = sys.argv[1]
    channel = sys.argv[2]
    gpu_idx = sys.argv[3]

    # Set the environmnent
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

    # Build and train the model
    ate = AutoEncoder(channel, storage=storage)
    ate.learn()