# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse, warnings

from package.dl_model import *

# Main algorithm

if __name__ == '__main__':

    warnings.simplefilter('ignore')

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-d', '--database', help='Initial Input Data')
    prs.add_argument('-c', '--channels', help='Channels Definition', nargs='*')
    # Parse the arguments
    prs = prs.parse_args()

    # Generate the corresponding channels
    dic = generate_channels(prs.channels)

    # Build and launch the corresponding DL model
    print('\n# Model launching:')
    time.sleep(0.5)
    mod = DL_Model(prs.database, dic)
    mod.learn()
    mod.write_to_file()


