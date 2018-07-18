# DINDIN Meryll
# July 18th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.ml_model import *

from hyperband.optimizer import *

# Defines the stacker for output consideration

class ML_Stacker:

    # Initialization of the stacker
    # models refers to which models to use (for diversity)
    # cv_folds refers to the amount of folds in cross-validation
    # mp is used by hyperband for multi-armed bandit
    # threads refers to the amount of concurrent threads
    def __init__(self, models, cv_folds=5, mp=False, threads=multiprocessing.cpu_count()):

        self.pbs, self.prd = [], []
        self.kfs = StratifiedKFold(n_splits=cv_folds, shuffle=True)
        self.threads = threads
        self.mp = mp

        self.lab = pd.read_csv('./dataset/label.csv', sep=';', index_col=0).values.ravel()
        self.n_c = len(np.unique(self.lab))

        for model in models:

            self.pbs.append(np.load('./models/PRB_{}.npy'.format(model)))
            self.prd.append(np.load('./models/PRD_{}.npy'.format(model)))

        self.pbs = np.hstack(tuple(self.pbs))
        self.prd = np.hstack(tuple(self.prd))

        sts = StandardScaler(with_std=False)
        sts.fit(np.vstack((self.pbs, self.prd)))
        self.pbs = sts.transform(self.pbs)
        self.prd = sts.transform(self.prd)
        self.out = np.zeros((len(self.prd), self.n_c))
            
    # Launched the hyperband optimization on meta-estimator
    # nme defines the type of estimator to use
    # max_iter put a threshold on the amount of hyperband iterations
    # log_file refers to where to write the scoring outputs
    def run(self, nme, max_iter=100, log_file='./models/CV_STACKING.log'):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        for idx, (i_t, i_e) in enumerate(self.kfs.split(self.lab, self.lab)):

            # Build the corresponding tuned model
            mkr = 'SK_{}'.format(idx)
            mod = ML_Model(threads=self.threads, mp=self.mp)
            mod.l_t = self.lab[i_t]
            mod.l_e = self.lab[i_e]
            mod.train = self.pbs[i_t]
            mod.valid = self.pbs[i_e]
            # Launch the hyperband optimization
            mod.learn(nme, marker=mkr, max_iter=max_iter)
            # Retrieve the scores
            a,k = mod.score(nme, marker=mkr)

            # Add the probabilities to the main launcher
            clf = joblib.load('./models/{}_{}.pk'.format(nme, mkr))
            self.out += clf.predict_proba(self.prd)

            # LOG file for those scores
            with open(log_file, 'a') as raw:
                raw.write('# CV_ROUND {} | Accuracy {:3f} | Kappa {:3f} \n'.format(idx, a, k))

            # Memory efficiency
            del mkr, mod, a, k

    # Write validation to file
    # out refers to the output path
    def write_to_file(self, out=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Compute the predictions for validation
        prd = np.asarray([np.argmax(ele) for ele in self.out])
        idx = np.arange(43830, 64422)
        res = np.hstack((idx.reshape(-1,1), prd.reshape(-1,1)))

        # Creates the relative dataframe
        res = pd.DataFrame(res, columns=['id', 'label'])

        # Write to csv
        if out is None: out = './results/cSTK_{}.csv'.format(int(time.time()))
        res.to_csv(out, index=False, header=True, sep=';')
