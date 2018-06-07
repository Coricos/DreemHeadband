# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from hyperband.optimizer import *

# Defines a structure for the machine-learning models

class ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # threads refers to the amount of affordable threads
    def __init__(self, path, threads=multiprocessing.cpu_count()):

        # Attributes
        self.input = path
        self.njobs = threads

        # Apply on the data
        with h5py.File(self.input, 'r') as dtb:
            # Load the labels and initialize training and testing sets
            self.l_t = dtb['lab_t'].value.ravel()
            self.l_e = dtb['lab_e'].value.ravel()
            # Define the specific anomaly issue
            self.n_c = len(np.unique(list(self.l_t) + list(self.l_e)))
            # Defines the vectors
            self.train = np.hstack((dtb['pca_t'].value, dtb['fea_t'].value))
            self.valid = np.hstack((dtb['pca_e'].value, dtb['fea_e'].value))
            self.evals = np.hstack((dtb['pca_v'].value, dtb['fea_v'].value))

        # Defines the different folds on which to apply the Hyperband
        self.folds = KFold(n_splits=5, shuffle=True)

    # Application of the ML models
    # nme refers to the type of model to use
    # marker refers to the identity of the model
    # max_iter refers to the amount of iterations with the hyperband algorithm
    def learn(self, nme, marker=None, max_iter=100):

        # Defines the data representation folder
        val = dict()
        val['x_train'] = self.train
        val['y_train'] = self.l_t
        val['w_train'] = sample_weight(self.l_t)
        val['x_valid'] = self.valid
        val['y_valid'] = self.l_e
        val['w_valid'] = sample_weight(self.l_e)
        # Defines the random search through cross-validation
        hyp = Hyperband(get_params, try_params, max_iter=max_iter, n_jobs=self.njobs)
        res = hyp.run(nme, val, skip_last=1)
        res = sorted(res, key = lambda x: x['kappa'])[0]
        # Extract the best estimator
        if nme == 'RFS':
            mod = RandomForestClassifier(**res['params'])
        if nme == 'GBT':
            mod = GradientBoostingClassifier(**res['params'])
        if nme == 'LGB':
            mod = lgb.LGBMClassifier(objective='multiclass', **res['params'])
        if nme == 'XGB':
            mod = xgb.XGBClassifier(**res['params'])
        if nme == 'SGD':
            mod = SGDClassifier(**res['params'])
        # Refit the best model
        mod.fit(val['x_train'], val['y_train'], sample_weight=val['w_train'])
        # Serialize the best obtained model
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        joblib.dump(mod, self.mod)

    # Defines the confusion matrix on train, test and validation sets
    # nme refers to a new path if necessary
    # marker allows specific redirection
    def confusion_matrix(self, nme, marker=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        clf = joblib.load(self.mod)

        # Method to build and display the confusion matrix
        def build_matrix(prd, true, title):

            lab = np.unique(list(prd) + list(true))
            cfm = confusion_matrix(true, prd)
            cfm = pd.DataFrame(cfm, index=lab, columns=lab)

            fig = plt.figure(figsize=(18,6))
            htp = sns.heatmap(cfm, annot=True, fmt='d', linewidths=1.)
            pth = self.out.split('/')[-1]
            acc = accuracy_score(true, prd)
            kap = kappa_score(true, prd)
            tle = '{} | {} | Accuracy: {:.2%} | Kappa: {:.2%}'
            plt.title(tle.format(title, pth, acc, kap))
            htp.yaxis.set_ticklabels(htp.yaxis.get_ticklabels(), 
                rotation=0, ha='right', fontsize=12)
            htp.xaxis.set_ticklabels(htp.xaxis.get_ticklabels(), 
                rotation=45, ha='right', fontsize=12)
            plt.ylabel('True label') 
            plt.xlabel('Predicted label')
            plt.show()

        # Compute the predictions for validation
        prd = clf.predict(self.valid)
        build_matrix(prd, self.l_e, 'TEST')
        del prd

    # Write validation to file
    # out refers to the output path
    # nme refers to a new path if necessary
    # marker allows specific redirection
    def write_to_file(self, nme, out=None, marker=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        clf = joblib.load(self.mod)

        # Compute the predictions for validation
        prd = clf.predict(self.evals)
        idx = np.arange(43830, 64422)
        res = np.hstack((idx.reshape(-1,1), prd.reshape(-1,1)))

        # Creates the relative dataframe
        res = pd.DataFrame(res, columns=['id', 'label'])

        # Write to csv
        if out is None: out = './results/test_{}.csv'.format(int(time.time()))
        res.to_csv(out, index=False, header=True, sep=';')
