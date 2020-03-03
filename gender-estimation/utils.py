# Author:  Meryll Dindin
# Date:    02 March 2020
# Project: DreemEEG

from imports import *

def stringify(x): 

	return x.astype(np.float32).tostring()

def featurize(vector, frequency): 
    
    return Featurize_1D(vector, sampling_frequency=frequency).getFeatures()
