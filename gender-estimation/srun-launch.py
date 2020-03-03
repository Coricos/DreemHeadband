# Author:  Meryll Dindin
# Date:    02 March 2020
# Project: DreemEEG

import os
import sys
import json

if __name__ == '__main__':

	# Load the slurm relative configuration
    with open('srun-config.json', 'r') as raw: cfg = json.load(raw)

    # Defines the command chunks
    cmd = ['nohup srun']
    for k, v in cfg.items(): cmd += ['--{}={}'.format(k, v)]
    cmd += sys.argv[1:]
    cmd += ['&']

    print(' '.join(cmd))
