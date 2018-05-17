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

from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Graphical imports

try: 
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gs
except:
	pass