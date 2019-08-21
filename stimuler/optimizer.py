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
        f_0 = pd.read_parquet('/'.join([directory, 'valid_cmp.pq']))
        f_1 = pd.read_parquet('/'.join([directory, 'valid_fea.pq']))
        f_0.index = f_1.index
        # Use as attributes
        self.x_v = f_0.join(f_1, how='left')
        # Match columns
        self.x_v = self.x_v[self.x_t.columns]

        # Memory efficiency
        del lab, f_0, f_1

        # Embedded methods
        self._categorize()

    def _categorize(self):

        # Reapply binary categorization
        self.x_t['sleep_stage'] = (self.x_t['sleep_stage'] < 0).astype('int')
        self.x_v['sleep_stage'] = (self.x_v['sleep_stage'] < 0).astype('int')

    def split(self, test_size=0.2, random_state=42, shuffle=True):

        arg = {'test_size': test_size, 'random_state': random_state, 'shuffle': shuffle}
        return self._preprocess(*train_test_split(self.x_t, self.y_t, **arg))

    def _preprocess(self, x_t, x_v, y_t, y_v):

        # Initialize scaler
        sts = StandardScaler()
        # Apply to train and test
        x_t = sts.fit_transform(x_t)
        x_v = sts.transform(x_v)
        # Apply to validation
        arg = {'columns': self.x_v.columns, 'index': self.x_v.index}
        self.out = pd.DataFrame(sts.transform(self.x_v), **arg)

        return x_t, x_v, y_t, y_v

class Experiment:

    _INIT = 50
    _OPTI = 0

    def __init__(self, name=str(int(time.time()))):

        self._id = name
        self.obj = 'classification'
        self.dir = '../experiments/{}'.format(self._id)
        if not os.path.exists(self.dir): os.mkdir(self.dir)
        self.log = Logger('/'.join([self.dir, 'logs.log']))
        self.dtb = DataLoader()

    def single(self, model, test_size=0.2, random_state=42, threads=cpu_count(), weights=False):

        self.log.info('Launch training for {} model'.format(model))
        self.log.info('Use {} concurrent threads\n'.format(threads))

        # Split the data for validation
        x_t, x_v, y_t, y_v = self.dtb.split(test_size=test_size, random_state=random_state)
        # Defines the problem
        prb = Prototype(x_t, x_v, y_t, y_v, model, self.obj, 'acc', threads=threads, weights=weights)
        # Launch the Bayesian optimization
        opt = Bayesian(prb, prb.loadBoundaries(), self.log, seed=random_state)
        opt.run(n_init=self._INIT, n_iter=self._OPTI)

        # Serialize the configuration file
        cfg = {'strategy': 'single', 'model': model, 'id': self._id, 'optimization': 'bayesian'}
        cfg.update({'random_state': random_state, 'threads': threads, 'test_size': test_size})
        cfg.update({'trial_init': self._INIT, 'trial_opti': self._OPTI})
        cfg.update({'best_score': prb.bestScore(), 'validation_metric': 'acc'})
        nme = '/'.join([self.dir, 'config.json'])
        with open(nme, 'w') as raw: json.dump(cfg, raw, indent=4, sort_keys=True)

        # Serialize parameters
        prm = prb.bestParameters()
        nme = '/'.join([self.dir, 'params.json'.format(model)])
        with open(nme, 'w') as raw: json.dump(prm, raw, indent=4, sort_keys=True)

    def saveModel(self):

        # Extract the parameters
        with open('/'.join([self.dir, 'config.json']), 'r') as raw: cfg = json.load(raw)
        with open('/'.join([self.dir, 'params.json']) , 'r') as raw: prm = json.load(raw)
        # Split the data for validation
        arg = {'test_size': cfg['test_size'], 'random_state': cfg['random_state']}
        x_t, x_v, y_t, y_v = self.dtb.split(**arg, shuffle=True)
        # Defines the problem
        arg = {'threads': cfg['threads']}
        prb = Prototype(x_t, x_v, y_t, y_v, cfg['model'], self.obj, 'acc', **arg)
        prb = prb.fitModel(prm, cfg['random_state'])

        # Serialize the model
        joblib.dump(prb, '/'.join([self.dir, 'model.jb']))

    def getModel(self):

        return joblib.load('/'.join([self.dir, 'model.jb']))

    def evaluateModel(self, model=None):

        if model is None: model = joblib.load('/'.join([self.dir, 'model.jb']))

        # Extract the parameters
        with open('/'.join([self.dir, 'config.json']), 'r') as raw: cfg = json.load(raw)
        with open('/'.join([self.dir, 'params.json']) , 'r') as raw: prm = json.load(raw)
        # Split the data for validation
        arg = {'test_size': cfg['test_size'], 'random_state': cfg['random_state']}
        _, x_v, _, y_v = self.dtb.split(**arg, shuffle=True)

        y_p = model.predict(x_v)
        lab = ['accuracy', 'f1 score', 'precision', 'recall', 'kappa']
        sco = np.asarray([
            accuracy_score(y_v, y_p),
            f1_score(y_v, y_p, average='weighted'),
            precision_score(y_v, y_p, average='weighted'),
            recall_score(y_v, y_p, average='weighted'),
            cohen_kappa_score(y_v, y_p)])
        cfm = confusion_matrix(y_v, y_p)

        plt.figure(figsize=(18,4))
        grd = gridspec.GridSpec(1, 3)

        arg = {'y': 1.05, 'fontsize': 14}
        plt.suptitle('General Classification Performances for Experiment {}'.format(self._id), **arg)

        ax0 = plt.subplot(grd[0, 0])
        crs = cm.Greens(sco)
        plt.bar(np.arange(len(sco)), sco, width=0.4, color=crs)
        for i,s in enumerate(sco): plt.text(i-0.15, s-0.05, '{:1.2f}'.format(s))
        plt.xticks(np.arange(len(sco)), lab)
        plt.xlabel('metric')
        plt.ylabel('percentage')

        ax1 = plt.subplot(grd[0, 1:])
        sns.heatmap(cfm, annot=True, fmt='d', axes=ax1, cbar=False, cmap="Greens")
        plt.ylabel('y_true')
        plt.xlabel('y_pred')

        plt.tight_layout()
        plt.show()

    def getImportances(self, model=None, n_display=30):

        if model is None: model = joblib.load('/'.join([self.dir, 'model.jb']))

        imp = model.feature_importances_ / np.sum(model.feature_importances_)
        imp = pd.DataFrame(np.vstack((self.dtb.out.columns, imp)).T, columns=['feature', 'importance'])
        imp = imp.sort_values(by='importance', ascending=False)
        imp = imp[:n_display]

        # Set the style of the axes and the text color
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333F4B'
        plt.rcParams['ytick.color'] = '#333F4B'
        plt.rcParams['text.color'] = '#333F4B'

        # Numeric placeholder for the y axis
        rge = list(range(1, len(imp.index)+1))

        fig, ax = plt.subplots(figsize=(18,10))
        # Create for each feature an horizontal line 
        plt.hlines(y=rge, xmin=0, xmax=imp.importance, color='salmon', alpha=0.4, linewidth=5)
        # Create for each feature a dot at the level of the percentage value
        plt.plot(imp.importance, rge, "o", markersize=5, color='red', alpha=0.3)

        # Set labels
        ax.set_xlabel('importance', fontsize=14, fontweight='black', color = '#333F4B')
        ax.set_ylabel('')
        # Set axis
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.yticks(rge, imp.feature)
        # Change the style of the axis spines
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        # Set the spines position
        ax.spines['bottom'].set_position(('axes', -0.04))
        ax.spines['left'].set_position(('axes', 0.015))
        plt.show()

    def submit(self, model=None):

        if model is None: model = joblib.load('/'.join([self.dir, 'model.jb']))
        y_p = model.predict(self.dtb.out.values)
        y_p = pd.DataFrame(np.vstack((self.dtb.out.index, y_p)).T, columns=['id', 'label'])
        y_p = y_p.set_index('id')
        y_p.to_csv('/'.join([self.dir, 'predictions.csv']))

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()    
    prs.add_argument('-m', '--mod', help='ModelType', type=str, default='LGB')
    prs.add_argument('-s', '--sze', help='TestSizes', type=float, default=0.33)
    prs.add_argument('-r', '--rnd', help='RandomSte', type=int, default=42)
    prs.add_argument('-c', '--cpu', help='NumOfCpus', type=int, default=cpu_count())
    prs.add_argument('-w', '--wei', help='UseWeight', type=bool, default=False)
    prs = prs.parse_args()

    exp = Experiment()
    # Run a single-shot learning
    exp.single(prs.mod, test_size=prs.sze, random_state=prs.rnd, threads=prs.cpu, weights=prs.wei)
    exp.saveModel()
    # Keep the results
    exp.submit()