# Author:  DINDIN Meryll
# Date:    05 August 2019
# Project: DreemHeadband

try: from stimuler.imports import *
except: from imports import *

# Challenger Package
from featurizers import Featurize_1D

class Featurizer:
    
    def __init__(self, sampling_frequency, max_workers=cpu_count()):
        
        self.cpu = max_workers
        self.frq = sampling_frequency
        
    def featurize_signal(self, signal):
        
        return Featurize_1D(signal, sampling_frequency=self.frq).getFeatures()
    
    def compute(self, signals):
        
        # Multiprocessed computation
        pol = Pool(processes=self.cpu)
        res = list(pol.map(self.featurize_signal, signals))
        pol.close()
        pol.join()
        # Concatenation
        res = pd.concat(res, axis=0)
        res.columns = [str(e) for e in res.columns]
        
        return res

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()    
    prs.add_argument('-f', '--frq', help='Frequency', type=int, default=125)
    prs.add_argument('-d', '--dir', help='Directory', type=str, default='train')
    prs.add_argument('-c', '--cpu', help='NumOfCpus', type=int, default=cpu_count())
    prs = prs.parse_args()

    # Run the featurization
    with h5py.File('../data/slow_waves/{}.h5'.format(prs.dir)) as dtb: sig = dtb['features'][:,11:]
    dtf = Featurizer(prs.frq, max_workers=prs.cpu).compute(sig)
    # Serialize to parquet format
    dtf.to_parquet('../data/slow_waves/{}_cmp.pq'.format(prs.dir))
