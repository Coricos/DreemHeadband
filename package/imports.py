# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

# Core packages

import h5py, multiprocessing, tqdm, nolds, sys
import pickle, warnings, time, pywt, joblib

import numpy as np
import pandas as pd
import scipy.signal as sg

from math import log, ceil
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from functools import partial
from collections import Counter
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import RandomOverSampler

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf

from keras import backend as K
from keras import regularizers
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Conv1D, Input, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import AveragePooling1D, AveragePooling2D, UpSampling1D
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Add
from keras.layers import GlobalMaxPooling1D, MaxoutDense, PReLU, LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adadelta, Adam
from keras.constraints import max_norm
from keras.layers.merge import concatenate
from keras.engine.topology import Layer
from keras.utils.training_utils import multi_gpu_model

# Graphical imports

try: 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gd
except:
    pass

# TDA packages

try:
    # Add gudhi to the environment path
    gpath = '2018-01-31-09-25-53_GUDHI_2.1.0/build/cython'
    build = '/home/intern/Downloads/{}'.format(gpath)
    sys.path.append(build)
    build = '/home/meryll/Documents/Environment/{}'.format(gpath)
    sys.path.append(build)
    del build, gpath

    import gudhi

except:
    pass