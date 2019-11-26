# Author:  DINDIN Meryll
# Date:    06 August 2019
# Project: DreemHeadband

try: from stimuler.imports import *
except: from imports import *

# Challenger Package
from optimizers import WrapperCV

class DataLoader:

    def __init__(self, directory='../data/slow_waves'):

        # Train dataset
        lab = pd.read_csv('/'.join([directory, 'label.csv']), index_col=0)
        f_0 = pd.read_parquet('/'.join([directory, 'train_cmp.pq']))
        f_1 = pd.read_parquet('/'.join([directory, 'train_fea.pq'])).drop('label', axis=1)
        f_0.index = f_1.index
        # Use as attributes
        self.x_t = f_0.join(f_1, how='left')
        self.y_t = lab.values.ravel()

        # Test dataset
        f_0 = pd.read_parquet('/'.join([directory, 'valid_cmp.pq']))
        f_1 = pd.read_parquet('/'.join([directory, 'valid_fea.pq']))
        f_0.index = f_1.index
        # Use as attributes
        self.x_v = f_0.join(f_1, how='left')
        # Match columns
        self.x_v = self.x_v[self.x_t.columns]

        # Remove irrelevant columns
        vtc = VarianceThreshold()
        self.x_t = vtc.fit_transform(self.x_t)
        self.x_v = vtc.transform(self.x_v)

        # Memory efficiency
        del lab, f_0, f_1

    def _binarize(self):

        # Reapply binary categorization
        self.x_t['sleep_stage'] = (self.x_t['sleep_stage'] < 0).astype('int')
        self.x_v['sleep_stage'] = (self.x_v['sleep_stage'] < 0).astype('int')

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()    
    prs.add_argument('-m', '--mod', help='ModelType', type=str, default='LGB')
    prs.add_argument('-r', '--rnd', help='RandomSte', type=int, default=12)
    prs.add_argument('-i', '--itr', help='NumTrials', type=int, default=80)
    prs.add_argument('-c', '--cpu', help='NumOfCpus', type=int, default=cpu_count())
    prs = prs.parse_args()

    # Preprocess the data
    dtb = DataLoader()
    # Specify the cv tools
    cfg = {'OPTUNA_TRIALS': prs.itr}
    pip = Pipeline([('mms', MinMaxScaler()), ('sts', StandardScaler())])
    # Launch the cross-validation
    prb = WrapperCV(dtb.x_t, dtb.x_v, dtb.y_t, folds=5, random_state=prs.rnd)
    prb.run(prs.mod, 'classification', 'acc', pip, threads=prs.cpu, weights=prs.wei, config=cfg)
