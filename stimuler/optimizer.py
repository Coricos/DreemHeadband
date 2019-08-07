# Author:  DINDIN Meryll
# Date:    06 August 2019
# Project: DreemHeadband

try: from stimuler.imports import *
except: from imports import *
# Challenger Package
from optimizers import Prototype, Bayesian, Logger

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
        f_0 = pd.read_parquet('/'.join([directory, 'train_cmp.pq']))
        f_1 = pd.read_parquet('/'.join([directory, 'train_fea.pq'])).drop('label', axis=1)
        f_0.index = f_1.index
        # Use as attributes
        self.x_v = f_0.join(f_1, how='left')

        # Memory efficiency
        del lab, f_0, f_1

        # Embedded methods
        self._preprocess()
        self._categorize()

    def _preprocess(self):

        # Initialize scaler
        sts = StandardScaler()
        # Apply to train and test
        x_t = sts.fit_transform(self.x_t)
        self.x_t = pd.DataFrame(x_t, columns=self.x_t.columns, index=self.x_t.index)
        x_v = sts.transform(self.x_v)
        self.x_v = pd.DataFrame(x_v, columns=self.x_v.columns, index=self.x_v.index)
        # Memory efficiency
        del sts, x_t, x_v

    def _categorize(self):

        # Reapply binary categorization
        self.x_t['sleep_stage'] = (self.x_t['sleep_stage'] < 0).astype('int')
        self.x_v['sleep_stage'] = (self.x_v['sleep_stage'] < 0).astype('int')

    def split(self, test_size=0.2, random_state=42, shuffle=True):

        arg = {'test_size': test_size, 'random_state': random_state, 'shuffle': shuffle}
        return train_test_split(self.x_t, self.y_t, **arg)

class Experiment:

    _INIT = 50
    _OPTI = 20

    def __init__(self, name=str(int(time.time()))):

        self._id = name
        self.obj = 'classification'
        self.dir = '../experiments/{}'.format(self._id)
        os.mkdir(self.dir)
        self.log = Logger('/'.join([self.dir, 'logs.log']))
        self.dtb = DataLoader()

    def single(self, model, test_size=0.2, random_state=42, threads=cpu_count()):

        self.log.info('Launch training for {} model'.format(model))
        self.log.info('Use {} concurrent threads\n'.format(threads))

        # Split the data for validation
        x_t, x_v, y_t, y_v = self.dtb.split(test_size=test_size, random_state=random_state)
        # Defines the problem
        prb = Prototype(x_t, x_v, y_t, y_v, model, self.obj, 'acc', threads=threads)
        # Launch the Bayesian optimization
        opt = Bayesian(prb, prb.loadBoundaries(), self.log, seed=random_state)
        opt.run(n_init=self._INIT, n_iter=self._OPTI)

        # Serialize the configuration file
        cfg = {'strategy': 'single', 'model': model, 'id': self._id, 'optimization': 'bayesian'}
        cfg.update({'random_state': random_state, 'threads': threads})
        cfg.update({'trial_init': self._INIT, 'trial_opti': self._OPTI})
        cfg.update({'best_score': prb.bestScore(), 'validation_metric': 'acc'})
        nme = '/'.join([self.dir, 'config.json'])
        with open(nme, 'w') as raw: json.dump(cfg, raw, indent=4, sort_keys=True)

        # Serialize parameters
        prm = prb.bestParameters()
        nme = '/'.join([self.dir, 'params.json'.format(model)])
        with open(nme, 'w') as raw: json.dump(prm, raw, indent=4, sort_keys=True)

if __name__ == '__main__':

    Experiment().single('XGB', test_size=0.33)