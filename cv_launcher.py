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
    # Channels acceleration
    prs.add_argument('--acc_cv2', help='Acceleration | Method CONV2D', type=str, default='F')
    prs.add_argument('--acc_cv1', help='Acceleration | Method CONV1D', type=str, default='F')
    prs.add_argument('--acc_cvl', help='Acceleration | Method CVLSTM', type=str, default='F')
    # Channels norm acceleration
    prs.add_argument('--n_a_cv1', help='Norm Acceleration | Method CONV1D', type=str, default='F')
    prs.add_argument('--n_a_cvl', help='Norm Acceleration | Method CVLSTM', type=str, default='F')
    # Channels electroencephalograms
    prs.add_argument('--eeg_cv2', help='Electroencephalograms | Method CONV2D', type=str, default='F')
    prs.add_argument('--eeg_cv1', help='Electroencephalograms | Method CONV1D', type=str, default='F')
    prs.add_argument('--eeg_cvl', help='Electroencephalograms | Method CVLSTM', type=str, default='F')
    prs.add_argument('--eeg_tda', help='Electroencephalograms | Method TDACV1', type=str, default='F')
    prs.add_argument('--eeg_enc', help='Electroencephalograms | Method ENCODE', type=str, default='F')
    prs.add_argument('--eeg_ate', help='Electroencephalograms | Method AUTOEN', type=str, default='F')
    prs.add_argument('--eeg_l_0', help='Electroencephalograms | Method LDCSIL', type=str, default='F')
    prs.add_argument('--eeg_l_1', help='Electroencephalograms | Method LDCSIL', type=str, default='F')
    # Channels norm electroencephalograms
    prs.add_argument('--n_e_cv1', help='Norm EEG | Method CONV1D', type=str, default='F')
    prs.add_argument('--n_e_cvl', help='Norm EEG | Method CVLSTM', type=str, default='F')
    # Channels oxygen measurements
    prs.add_argument('--por_cv1', help='Oxymeter | Method CONV1D', type=str, default='F')
    prs.add_argument('--por_cvl', help='Oxymeter | Method CVLSTM', type=str, default='F')
    prs.add_argument('--por_enc', help='Oxymeter | Method ENCODE', type=str, default='F')
    prs.add_argument('--por_ate', help='Oxymeter | Method AUTOEN', type=str, default='F')
    prs.add_argument('--poi_cv1', help='Oxymeter | Method CONV1D', type=str, default='F')
    prs.add_argument('--poi_cvl', help='Oxymeter | Method CVLSTM', type=str, default='F')
    prs.add_argument('--poi_enc', help='Oxymeter | Method ENCODE', type=str, default='F')
    prs.add_argument('--poi_ate', help='Oxymeter | Method AUTOEN', type=str, default='F')
    # Basic channels
    prs.add_argument('--feature', help='Features | Method DENSE', type=str, default='T')
    # Parse the arguments
    prs = prs.parse_args()

    # Define the corresponding channels
    dic = {
           'with_acc_cv2': prs.acc_cv2 == 'T',
           'with_acc_cv1': prs.acc_cv1 == 'T',
           'with_acc_cvl': prs.acc_cvl == 'T',
           'with_n_a_cv1': prs.n_a_cv1 == 'T',
           'with_n_a_cvl': prs.n_a_cvl == 'T',
           'with_eeg_cv2': prs.eeg_cv2 == 'T',
           'with_eeg_cv1': prs.eeg_cv1 == 'T',
           'with_eeg_cvl': prs.eeg_cvl == 'T',
           'with_eeg_tda': prs.eeg_tda == 'T',
           'with_eeg_enc': prs.eeg_enc == 'T',
           'with_eeg_ate': prs.eeg_ate == 'T',
           'with_eeg_l_0': prs.eeg_l_0 == 'T',
           'with_eeg_l_1': prs.eeg_l_1 == 'T',
           'with_n_e_cv1': prs.n_e_cv1 == 'T',
           'with_n_e_cvl': prs.n_e_cvl == 'T',
           'with_por_cv1': prs.por_cv1 == 'T',
           'with_por_cvl': prs.por_cvl == 'T',
           'with_por_enc': prs.por_enc == 'T',
           'with_por_ate': prs.por_ate == 'T',
           'with_poi_cv1': prs.poi_cv1 == 'T',
           'with_poi_cvl': prs.poi_cvl == 'T',
           'with_poi_enc': prs.poi_enc == 'T',
           'with_poi_ate': prs.poi_ate == 'T',
           'with_fea': prs.feature == 'T',
           }

    # Launch cross-validation for ml models
    if prs.objectif == 'ml':
        mod = CV_ML_Model('./dataset/sca_train.h5', k_fold=prs.folds, threads=prs.threads)
        mod.launch(prs.name, log_file=prs.log_file)

    # Launch cross-validation for dl models
    if prs.objectif == 'dl':
        mod = CV_DL_Model(dic, storage='./dataset', n_iter=prs.folds)
        mod.launch()


