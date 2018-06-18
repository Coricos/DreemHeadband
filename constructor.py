# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse, warnings

from package.database import *

# Pipeline description of dataset construction

def build_dataset(storage, label_csv, output):

    dtb = Database(storage=storage)
    dtb.build(out_storage=storage)
    dtb.load_labels(label_csv)
    dtb.add_norm_acc()
    dtb.add_norm_eeg()
    dtb.add_features()
    dtb.add_paper_features()
    dtb.add_betti_curves()
    dtb.rescale()
    dtb.preprocess(output)

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-s', '--storage', help='Path to Storage',
                     type=str, default='./dataset')
    prs.add_argument('-l', '--ann_csv', help='Input File for Labels',
                     type=str, default='./dataset/label.csv')
    prs.add_argument('-t', '--towards', help='Output Directory File',
                     type=str, default='./dataset/DTB_Headband.h5')
    # Parse the arguments
    prs = prs.parse_args()
    
    warnings.simplefilter('ignore')

    build_dataset(prs.storage, prs.ann_csv, prs.towards)