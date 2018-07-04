# Author : Dindin Meryll
# Date : 01/07/2018
# Dreem Headband Sleep Phases Classification Challenge

from cmaes.functions import *

# Main Launcher

if __name__ == '__main__':

	# Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yml', type=str, help='Configuration file in YAML format')
    parser = parser.parse_args()
    
    curio.run(amain(parser.config))