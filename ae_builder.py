# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse

from package.ds_model import *

# Launch the pipeline
if __name__ == '__main__':

    # Initialize the arguments
    test_rt = float(sys.argv[1])
    storage = sys.argv[2]
    channel = sys.argv[3]
    gpu_idx = sys.argv[4]

    # Set the environmnent
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

    # Build and train the model
    ate = AutoEncoder(channel, storage=storage)
    ate.learn(test_ratio=test_rt)