# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.toolbox import *

class Database:

    # Initialization
    # storage refers to where to get the datasets
    def __init__(self, storage='./dataset'):

        self.train_pth = '{}/train.h5'.format(storage)
        self.valid_pth = '{}/valid.h5'.format(storage)
        self.train_out = '{}/dts_train.h5'.format(storage)
        self.valid_out = '{}/dts_valid.h5'.format(storage)
        self.train_sca = '{}/sca_train.h5'.format(storage)
        self.valid_sca = '{}/sca_valid.h5'.format(storage)

        self.storage = storage
        self.sets_size = []
        
        with h5py.File(self.train_pth, 'r') as dtb:
            self.sets_size.append(dtb['po_r'].shape[0])
            self.keys = list(dtb.keys())
        with h5py.File(self.valid_pth, 'r') as dtb:
            self.sets_size.append(dtb['po_r'].shape[0])

    # Apply filtering and interpolation on the samples
    # sampling_freq refers to the desired sampling frequency
    # out_storage refers to where to put the newly build datasets
    def build(self, sampling_freq=100, out_storage='/mnt/Storage'):

        # Defines the parameters for each key
        fil = {'po_r': True, 'po_ir': True,
               'acc_x': False, 'acc_y': False, 'acc_z': False,
               'eeg_1': True, 'eeg_2': True, 'eeg_3': True, 'eeg_4': True}

        dic = {'eeg_1': (4, 20), 'eeg_2': (4, 20), 'eeg_3': (4, 20),
               'eeg_4': (4, 20), 'po_ir': (3, 5), 'po_r': (3, 5)}

        # Iterates over the keys
        for key in tqdm.tqdm(fil.keys()):
            # Link inputs to outputs
            for pth, out in zip([self.train_pth, self.valid_pth], 
                                [self.train_out, self.valid_out]):
                # Load the values
                with h5py.File(pth, 'r') as dtb: val = dtb[key].value

                # Apply the kalman filter if needed
                if fil[key]:
                    pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    arg = {'std_factor': dic[key][0], 'smooth_window': dic[key][1]}
                    fun = partial(kalman_filter, **arg)
                    val = np.asarray(pol.map(fun, val))
                    pol.close()
                    pol.join()

                # Adapt the size of the vectors
                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                fun = partial(interpolate, size=30*sampling_freq)
                val = np.asarray(pol.map(fun, val))
                pol.close()
                pol.join()

                # Serialize the outputs
                with h5py.File(out, 'a') as dtb:
                    if dtb.get(key): del dtb[key]
                    dtb.create_dataset(key, data=val)

                # Memory efficiency
                del pol, fun, val

    # Load the corresponding labels
    # input refers to the input_file in which the labels are stored
    def load_labels(self, input):

        lab = pd.read_csv(input, sep=';', index_col=0)

        with h5py.File(self.train_out, 'a') as dtb:
            # Serialize the labels
            if dtb.get('lab'): del dtb['lab']
            dtb.create_dataset('lab', data=lab.values)

    # Build the norm of the accelerometers
    def add_norm_acc(self):

        # Iterates over both the training and validation sets
        for pth in [self.train_out, self.valid_out]:

            with h5py.File(pth, 'r') as dtb:

                # Aggregates the values
                tmp = np.square(dtb['acc_x'].value)
                tmp += np.square(dtb['acc_y'].value)
                tmp += np.square(dtb['acc_z'].value)

            # Serialize the result
            with h5py.File(pth, 'a') as dtb:

                if dtb.get('norm_acc'): del dtb['norm_acc']
                dtb.create_dataset('norm_acc', data=np.sqrt(tmp))

            # Memory efficiency
            del tmp

    # Build the norm of the ECGs
    def add_norm_eeg(self):

        # Iterates over both the training and validation sets
        for pth in [self.train_out, self.valid_out]:

            with h5py.File(pth, 'r') as dtb:

                # Aggregates the values
                tmp = np.square(dtb['eeg_1'].value)
                tmp += np.square(dtb['eeg_2'].value)
                tmp += np.square(dtb['eeg_3'].value)
                tmp += np.square(dtb['eeg_4'].value)

            # Serialize the result
            with h5py.File(pth, 'a') as dtb:

                if dtb.get('norm_eeg'): del dtb['norm_eeg']
                dtb.create_dataset('norm_eeg', data=np.sqrt(tmp))

            # Memory efficiency
            del tmp

    # Build the corresponding Betti curves
    def add_betti_curves(self):

        lmt = []
        # Get the betti curves limitations
        for pth in [self.train_out, self.valid_out]:

            # Iterates over the EEGs signals
            for key in tqdm.tqdm(range(1, 5)):

                # Load the corresponding values
                with h5py.File(pth, 'r') as dtb: 
                    val = dtb['eeg_{}'.format(key)].value

                # Computes the persistent limits for the relative patient
                pol = multiprocessing.Pool(processes=self.threads)
                lmt.append(np.asarray(pol.map(persistent_limits, val)))
                pol.close()
                pol.join()
                # Memory efficiency
                del val, pol

        # Extracts the main limits
        lmt = np.vstack(tuple(lmt))
        mnu, mxu = min(lmt[:,0]), max(lmt[:,1]) 
        mnd, mxd = min(lmt[:,2]), max(lmt[:,3])
        # Memory efficiency
        del lmt

        # Serialize the obtained threshold
        with open('./tools/TDA_limits.pk', 'wb') as raw:
            dic = {'min_up': mnu, 'max_up': mxu, 'min_dw': mnd, 'max_dw': mxd}
            pickle.dump(dic, raw)
            del dic

        # Build the betti curves
        for pth in [self.train_out, self.valid_out]:

            # Iterates over the EEGs signals
            for key in tqdm.tqdm(range(1, 5)):

                # Load the corresponding values
                with h5py.File(pth, 'r') as dtb: 
                    val = dtb['eeg_{}'.format(key)].value
                    
                # Multiprocessed computation
                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                arg = {'mnu': mnu, 'mxu': mxu, 'mnd': mnd, 'mxd': mxd}
                fun = partial(compute_betti_curves, **arg)
                res = np.asarray(pol.map(fun, val))
                pol.close()
                pol.join()
                # Memory efficiency
                del val, pol, arg, fun

                # Serialize the output
                with h5py.File(pth, 'a') as dtb:
                    new = 'bup_{}'.format(key)
                    if dtb.get(new): del dtb[new]
                    dtb.create_dataset(new, data=res[:,0,:])
                    new = 'bdw_{}'.format(key)
                    if dtb.get(new): del dtb[new]
                    dtb.create_dataset(new, data=res[:,1,:])

    # Build the corresponding landscapes
    def add_landscapes(self):

        # Build the betti curves
        for pth in [self.train_out, self.valid_out]:

            res = []
            # Iterates over the EEGs signals
            for key in tqdm.tqdm(range(1, 5)):

                # Load the corresponding values
                with h5py.File(pth, 'r') as dtb: 
                    val = dtb['eeg_{}'.format(key)].value
                    
                # Multiprocessed computation
                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                res = np.asarray(pol.map(compute_landscapes, val))
                pol.close()
                pol.join()

                # Serialize the output
                with h5py.File(pth, 'a') as dtb:
                    new = 'l_0_{}'.format(key)
                    if dtb.get(new): del dtb[new]
                    dtb.create_dataset(new, data=res[:,:10,:])
                    new = 'l_1_{}'.format(key)
                    if dtb.get(new): del dtb[new]
                    dtb.create_dataset(new, data=res[:,10:,:])

    # Build the features for each channel
    # n_components refers to the PCA transformation
    def add_features(self, n_components=5):

        lst = ['norm_acc', 'po_ir', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']

        for pth in [self.train_out, self.valid_out]:

            res = []
            # Iterates over the keys
            for key in tqdm.tqdm(lst):

                # Load the corresponding values
                with h5py.File(pth, 'r') as dtb: val = dtb[key].value
                # Multiprocessed computation
                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                res.append(np.asarray(pol.map(compute_features, val)))
                pol.close()
                pol.join()

            # Serialize the output
            with h5py.File(pth, 'a') as dtb:
                if dtb.get('fea'): del dtb['fea']
                dtb.create_dataset('fea', data=np.hstack(tuple(res)))
                del res

        # Build the features relative to their PCA reduction
        train_pca, valid_pca = [], []

        # Iterates over the keys
        for key in tqdm.tqdm(lst):

            # Defines the PCA transform adapted to incremental learning
            pca = IncrementalPCA(n_components=n_components)
            # Partial fit over training and validation
            for pth in [self.train_out, self.valid_out]:
                with h5py.File(pth, 'r') as dtb:
                    pca.partial_fit(dtb[key].value)
            # Apply transformation on training set
            with h5py.File(self.train_out, 'r') as dtb:
                train_pca.append(pca.transform(dtb[key].value))
            # Apply transformation on validation set
            with h5py.File(self.valid_out, 'r') as dtb:
                valid_pca.append(pca.transform(dtb[key].value))

        # Serialization for the training results
        with h5py.File(self.train_out, 'a') as dtb:
            fea, pca = dtb['fea'].value, np.hstack(tuple(train_pca))
            del dtb['fea']
            dtb.create_dataset('fea', data=np.hstack((fea, pca)))
            del fea, pca
        # Serialization for the validation results
        with h5py.File(self.valid_out, 'a') as dtb:
            fea, pca = dtb['fea'].value, np.hstack(tuple(valid_pca))
            del dtb['fea']
            dtb.create_dataset('fea', data=np.hstack((fea, pca)))
            del fea, pca

        # Memory efficiency
        del train_pca, valid_pca, lst

    # Rescale the datasets considering both training and validation
    def rescale(self, size=400):

        with h5py.File(self.train_out, 'r') as dtb:
            res = ['norm_acc', 'norm_eeg', 'acc_x', 'acc_y', 'acc_z', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']
            unt = ['bup_1', 'bup_2', 'bup_3', 'bup_4', 'bdw_1', 'bdw_2', 'bdw_3', 'bdw_4']
            ldc = ['l_0_1', 'l_0_2', 'l_0_3', 'l_0_4', 'l_1_1', 'l_1_2', 'l_1_3', 'l_1_4']
            oth = [key for key in list(dtb.keys()) if key not in res + unt + ldc + ['lab']]

        # Transfer the labels from DTS to SCA

        with h5py.File(self.train_sca, 'a') as dtb:
            if dtb.get('lab'): del dtb['lab']
            with h5py.File(self.train_out, 'r') as inp:
                dtb.create_dataset('lab', data=inp['lab'].value)
        print('# Label transfer ...')
        time.sleep(0.5)
        # Specific scaling for the temporal signals
        print('# Rescaling temporal signals ...')
        time.sleep(0.5)
        for key in tqdm.tqdm(res):

            with h5py.File(self.train_out, 'r') as dtb: v_t = dtb[key].value
            with h5py.File(self.valid_out, 'r') as dtb: v_v = dtb[key].value

            old, m_x = [], max(np.max(np.abs(v_t)), np.max(np.abs(v_v)))
            coe = list(np.max(v_t, axis=1)) + list(np.max(v_v, axis=1))
            coe = np.median(np.asarray(coe))

            for inp in [self.train_out, self.valid_out]:

                with h5py.File(inp, 'r') as dtb:

                    pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    fun = partial(resize_time_serie, size=size, threshold=coe)
                    old += pol.map(fun, dtb[key].value)
                    pol.close()
                    pol.join()

            old = np.vstack(tuple(old))
            mms = MinMaxScaler(feature_range=(0,2))
            sts = StandardScaler(with_std=False)
            pip = Pipeline([('mms', mms), ('sts', sts)])
            old = pip.fit_transform(np.hstack(tuple(old)).reshape(-1,1)).reshape(old.shape)

            with h5py.File(self.train_sca, 'a') as dtb: 
                if dtb.get(key): del dtb[key]
                dtb.create_dataset(key, data=old[:len(v_t)])
            with h5py.File(self.valid_sca, 'a') as dtb:
                if dtb.get(key): del dtb[key]
                dtb.create_dataset(key, data=old[len(v_t):])
            
            # Memory efficiency
            del mms, sts, pip, old, v_t, v_v, m_x

        # Rescaling for the betti curves
        print('# Rescaling betti curves ...')
        time.sleep(0.5)
        for key in tqdm.tqdm(unt):

            # Defines the scalers
            mms = MinMaxScaler(feature_range=(0,1))

            for pth in [self.train_out, self.valid_out]:

                with h5py.File(pth, 'r') as dtb:
                    mms.partial_fit(np.hstack(dtb[key].value).reshape(-1,1))

            for inp, out in [(self.train_out, self.train_sca), (self.valid_out, self.valid_sca)]:

                with h5py.File(inp, 'r') as dtb:

                    shp = dtb[key].shape
                    tmp = np.hstack(dtb[key].value).reshape(-1,1)
                    res = mms.transform(tmp).reshape(shp)

                with h5py.File(out, 'a') as dtb:

                    if dtb.get(key): del dtb[key]
                    dtb.create_dataset(key, data=res)

            # Memory efficiency
            del mms, tmp

        # Rescaling for the persistent landscapes
        print('# Rescaling persistent landscapes ...')
        time.sleep(0.5)
        for key in tqdm.tqdm(ldc):

            m_x = []

            for pth in [self.train_out, self.valid_out]:
                # Defines the maximum value for all landscapes
                with h5py.File(pth, 'r') as dtb:
                    m_x.append(np.max(dtb[key].value))

            m_x = max(tuple(m_x))

            for inp, out in [(self.train_out, self.train_sca), (self.valid_out, self.valid_sca)]:

                with h5py.File(inp, 'r') as dtb:
                    val = dtb[key].value / m_x

                with h5py.File(out, 'a') as dtb:
                    if dtb.get(key): del dtb[key]
                    dtb.create_dataset(key, data=val)
                    del val

        # Specific scaling for features datasets
        print('# Rescaling other features ...')
        time.sleep(0.5)
        for key in tqdm.tqdm(oth):

            # Build the scaler
            mms = MinMaxScaler(feature_range=(-1, 0))
            sts = StandardScaler(with_std=False)

            for pth in [self.train_out, self.valid_out]:
                # Partial fit for both training and validation
                with h5py.File(pth, 'r') as dtb:
                    mms.partial_fit(remove_out_with_mean(dtb[key].value))

            for pth in [self.train_out, self.valid_out]:
                # Partial fit for both training and validation
                with h5py.File(pth, 'r') as dtb:
                    sts.partial_fit(mms.transform(remove_out_with_mean(dtb[key].value)))

            pip = Pipeline([('mms', mms), ('sts', sts)])

            for inp, out in [(self.train_out, self.train_sca), (self.valid_out, self.valid_sca)]:

                with h5py.File(inp, 'r') as dtb:
                    val = pip.transform(remove_out_with_mean(dtb[key].value))

                with h5py.File(out, 'a') as dtb:
                    if dtb.get(key): del dtb[key]
                    dtb.create_dataset(key, data=val)
                    del val

    # Defines a way to reduce the problem
    # output refers to where to serialize the output database
    # size refers to the amount of vectors to keep
    def truncate(self, output, size=3000):

        with h5py.File(self.train_sca, 'r') as inp:

            # Defines the indexes for extraction
            arg = {'size': size, 'replace': False}
            idx = np.random.choice(np.arange(inp['acc_x'].shape[0]), **arg)

            with h5py.File(output, 'a') as out:

                for key in tqdm.tqdm(list(inp.keys())):
                    # Iterated serialization of the key component
                    out.create_dataset(key, data=inp[key].value[idx])

    # Defines both training and testing instances
    # output refers to where to put the data
    # test refers to the test_size
    def preprocess(self, output, test=0.1):

        # Split the training set into both training and testing
        with h5py.File(self.train_sca, 'r') as dtb:

            idx = np.arange(dtb['lab'].shape[0])
            arg = {'test_size': test, 'shuffle': True}
            i_t, i_e, _, _ = train_test_split(idx, idx, **arg)
            i_t = shuffle(i_t)

            for key in tqdm.tqdm(list(dtb.keys())):

                with h5py.File(output, 'a') as out:

                    lab_t, lab_e = '{}_t'.format(key), '{}_e'.format(key)

                    if out.get(lab_t): del out[lab_t]
                    out.create_dataset(lab_t, data=dtb[key].value[i_t])
                    if out.get(lab_e): del out[lab_e]
                    out.create_dataset(lab_e, data=dtb[key].value[i_e])

        # Adds the validation set into the output database
        with h5py.File(self.valid_sca, 'r') as dtb:

            for key in tqdm.tqdm(list(dtb.keys())):

                with h5py.File(output, 'a') as out:

                    lab_v = '{}_v'.format(key)

                    if out.get(lab_v): del out[lab_v]
                    out.create_dataset(lab_v, data=dtb[key].value)
    
    # Build the multiple datasets necessary for cross-validation
    # folds refers to the amount of cross-validation rounds
    # storage refers to the root directory for datasets storage
    def build_cv(self, folds, storage='./dataset'):

        # Defines the cross-validation splits
        kfs = KFold(n_splits=folds)
        with h5py.File(self.train_sca, 'r') as dtb: 
            lab = dtb['lab'].value.ravel()

        # For each round, creates a new dataset
        for idx, (i_t, i_e) in enumerate(kfs.split(np.arange(len(lab)))):

            output = '{}/CV_ITER_{}.h5'.format(storage, idx)
            print('\n# Building CV_ITER_{}.h5'.format(idx))

            # Split the training set into both training and testing
            with h5py.File(self.train_sca, 'r') as dtb:

                print('# Train and Test ...')
                time.sleep(0.5)
                for key in tqdm.tqdm(list(dtb.keys())):

                    with h5py.File(output, 'a') as out:

                        key_t, key_e = '{}_t'.format(key), '{}_e'.format(key)

                        if out.get(key_t): del out[key_t]
                        out.create_dataset(key_t, data=dtb[key].value[i_t])
                        if out.get(key_e): del out[key_e]
                        out.create_dataset(key_e, data=dtb[key].value[i_e])

            # Adds the validation set into the output database
            with h5py.File(self.valid_sca, 'r') as dtb:

                print('# Validation ...')
                time.sleep(0.5)
                for key in tqdm.tqdm(list(dtb.keys())):

                    with h5py.File(output, 'a') as out:

                        key_v = '{}_v'.format(key)

                        if out.get(key_v): del out[key_v]
                        out.create_dataset(key_v, data=dtb[key].value)

