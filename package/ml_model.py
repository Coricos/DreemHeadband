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
    def __init__(self, path=None, threads=multiprocessing.cpu_count(), mp=False):

        # Attributes
        self.njobs = threads
        self.mp = mp

        if path:
            # Needed attribute
            self.input = path
            # Apply on the data
            with h5py.File(self.input, 'r') as dtb:
                # Load the labels and initialize training and testing sets
                self.l_t = dtb['lab_t'].value.ravel()
                self.l_e = dtb['lab_e'].value.ravel()
                # Define the specific anomaly issue
                self.n_c = len(np.unique(list(self.l_t) + list(self.l_e)))
                # Defines the vectors
                self.train = dtb['fea_t'].value
                self.valid = dtb['fea_e'].value

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
        hyp = Hyperband(get_params, try_params, max_iter=max_iter, n_jobs=self.njobs, mp=self.mp)
        res = hyp.run(nme, val, skip_last=1)
        res = sorted(res, key = lambda x: x['kappa'])[0]
        # Filter the best params
        params = res['params']
        if 'mp' in params.keys(): del params['mp']
        if 'n_mp' in params.keys(): del params['n_mp']
        # Extract the best estimator
        if nme == 'ETS':
            mod = ExtraTreesClassifier(**params)
        if nme == 'RFS':
            mod = RandomForestClassifier(n_jobs=self.njobs, **params)
        if nme == 'GBT':
            mod = GradientBoostingClassifier(**params)
        if nme == 'LGB':
            mod = lgb.LGBMClassifier(n_jobs=self.njobs, objective='multiclass', **params)
        if nme == 'XGB':
            mod = xgb.XGBClassifier(n_jobs=self.njobs, **params)
        if nme == 'SGD':
            mod = SGDClassifier(**params)
        # Refit the best model
        mod.fit(val['x_train'], val['y_train'], sample_weight=val['w_train'])
        # Serialize the best obtained model
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        joblib.dump(mod, self.mod)

    # Defines the confusion matrix on train, test and validation sets
    # nme refers to a new path if necessary
    # marker allows specific redirection
    def score(self, nme, marker=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        clf = joblib.load(self.mod)

        # Compute the predictions for validation
        prd = clf.predict(self.valid)

        return accuracy_score(self.l_e, prd), kappa_score(self.l_e, prd)

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
            pth = self.mod.split('/')[-1]
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

    # Get the probabilities from the testing set
    # nme refers to a new path if necessary
    # marker allows specific redirection
    def proba(self, nme, marker=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        clf = joblib.load(self.mod)

        # Compute the predictions for validation
        prb = clf.predict_proba(self.valid)

        return prb

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

        with h5py.File(self.input, 'r') as dtb: self.evals = dtb['fea_v'].value
        # Compute the predictions for validation
        prd = clf.predict(self.evals)
        idx = np.arange(43830, 64422)
        res = np.hstack((idx.reshape(-1,1), prd.reshape(-1,1)))

        # Creates the relative dataframe
        res = pd.DataFrame(res, columns=['id', 'label'])

        # Write to csv
        if out is None: out = './results/test_{}.csv'.format(int(time.time()))
        res.to_csv(out, index=False, header=True, sep=';')

# Defines a structure for a cross_validation

class CV_ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # k_fold refers to 
    # threads refers to the amount of affordable threads
    def __init__(self, path, k_fold=7, mp=False, threads=multiprocessing.cpu_count()):

        # Attributes
        self.input = path
        self.njobs = threads
        self.mp = mp

        # Apply on the data
        with h5py.File(self.input, 'r') as dtb:
            # Load the labels and initialize training and testing sets
            self.lab = dtb['lab'].value.ravel()
            # Define the specific anomaly issue
            self.n_c = len(np.unique(self.lab))
            # Defines the vectors
            self.vec = dtb['fea'].value[:,34:]

        # Defines the cross-validation splits
        self.kfs = StratifiedKFold(n_splits=k_fold, shuffle=True)

        # Apply feature filtering based on variance
        vtf = VarianceThreshold(threshold=0.0)
        self.vec = vtf.fit_transform(self.vec)
        joblib.dump(vtf, './models/VTF_Selection.jb')

    # CV Launcher
    # nme refers to the type of model to be launched
    # max_iter refers to the amount of iterations with the hyperband algorithm
    # log_file refers to where to store the intermediate scores
    def launch(self, nme, max_iter=100, log_file='./models/CV_SCORING.log'):

        out = np.zeros((len(self.lab), self.n_c))

        for idx, (i_t, i_e) in enumerate(self.kfs.split(self.lab, self.lab)):

            # Build the corresponding tuned model
            mkr = 'CV_{}'.format(idx)
            mod = ML_Model(threads=self.njobs, mp=self.mp)
            mod.l_t = self.lab[i_t]
            mod.l_e = self.lab[i_e]
            mod.train = self.vec[i_t]
            mod.valid = self.vec[i_e]
            # Launch the hyperband optimization
            mod.learn(nme, marker=mkr, max_iter=max_iter)
            # Retrieve the scores
            a,k = mod.score(nme, marker=mkr)
            # Add the probabilities to the main launcher
            out[i_e,:] = mod.proba(nme, marker=mkr)
            # LOG file for those scores
            with open(log_file, 'a') as raw:
                raw.write('# CV_ROUND {} | Accuracy {:3f} | Kappa {:3f} \n'.format(idx, a, k))

            # Memory efficiency
            del mkr, mod, a, k

        # Write the general score
        prd = [np.argmax(ele) for ele in out]
        acc = accuracy_score(self.lab, prd)
        kap = kappa_score(self.lab, prd)
        # LOG file for those scores
        with open(log_file, 'a') as raw:
            raw.write('\n')
            raw.write('# FINAL SCORE | Accuracy {:3f} | Kappa {:3f} \n'.format(acc, kap))
            raw.write('\n')

        # Serialize the probabilities as new features
        np.save('./models/PRB_{}.npy'.format(nme), out)

    # Stacking of predictions
    # valid refers to where the validation input is stored
    # nme refers to the name of the estimator
    # scaler refers whether feature extraction has been used
    def make_predictions(self, valid, nme, scaler=None):

        # Apply on the data
        with h5py.File(valid, 'r') as dtb:
            if scaler:
                # Load the scaler
                vtf = joblib.load(scaler)
                # Defines the vectors
                vec = vtf.transform(dtb['fea'].value[:,34:])
            else:
                vec = dtb['fea'].value[:,34:]

        # Initial vector for result storing
        res = np.zeros((len(vec), self.n_c))

        # Look for all available models
        lst = sorted(glob.glob('./models/{}_*.pk'.format(nme)))
        for mod in lst:
            # Load the model and make the predictions
            mod = joblib.load(mod)
            res += np_utils.to_categorical(mod.predict(vec), num_classes=self.n_c)

        # Write to file
        res = np.asarray([np.argmax(ele) for ele in res])
        idx = np.arange(43830, 64422)
        res = np.hstack((idx.reshape(-1,1), res.reshape(-1,1)))

        # Creates the relative dataframe
        res = pd.DataFrame(res, columns=['id', 'label'])

        # Write to csv
        out = './results/c{}_{}.csv'.format(nme, int(time.time()))
        res.to_csv(out, index=False, header=True, sep=';')



