# Author:  DINDIN Meryll
# Date:    05 August 2019
# Project: DreemHeadband

try: from stimuler.imports import *
except: from imports import *

def entropy(signal):
    
    cnt, dta = Counter(), np.round(signal, 4)
    for ele in dta: cnt[ele] += 1
    pbs = [signal / len(dta) for signal in cnt.values()]
    pbs = np.asarray([prd for prd in pbs if prd > 0.0])

    return - np.sum(pbs*np.log2(pbs))