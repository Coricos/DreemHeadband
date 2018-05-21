# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

# Core packages

import h5py, multiprocessing, tqdm, nolds

import numpy as np
import pandas as pd
import scipy.signal as sg

from functools import partial
from scipy.interpolate import interp1d

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras import backend as K
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Conv1D, Input, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling1D, MaxoutDense
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.merge import concatenate
from keras.engine.topology import Layer

# Graphical imports

try: 
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gs
except:
	pass