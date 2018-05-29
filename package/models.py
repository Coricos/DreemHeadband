# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from package.callback import *

# Defines a structure for the machine-learning models

class ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # threads refers to the amount of affordable threads
    def __init__(self, path, threads=multiprocessing.cpu_count()):

        # Attributes
        self.input = path
        self.njobs = threads
        self.train

        # Clearer definition
        msk_labels = list(np.unique(msk_labels))
        print('# Mask {} during training'.format(msk_labels))

        # Apply on the data
        with h5py.File(self.inp, 'r') as dtb:
            # Load the labels and initialize training and testing sets
            self.m_t = get_mask(dtb['t_lab'].value, lab_to_del=msk_labels)
            self.m_e = get_mask(dtb['e_lab'].value, lab_to_del=msk_labels)
            self.m_v = get_mask(dtb['v_lab'].value, lab_to_del=msk_labels)
            self.l_t = dtb['t_lab'].value[self.m_t]
            self.l_e = dtb['e_lab'].value[self.m_e]
            self.l_v = dtb['v_lab'].value[self.m_v]
            # Memory efficiency
            del msk_labels

        # Suppress the missing labels for categorical learning
        self.lbe = LabelEncoder()
        tmp = np.unique(list(self.l_t) + list(self.l_e) + list(self.l_v))
        self.lbe.fit(tmp)
        self.l_t = self.lbe.transform(self.l_t)
        self.l_e = self.lbe.transform(self.l_e)
        self.l_v = self.lbe.transform(self.l_v)
        # Define the specific anomaly issue
        self.num_classes = len(tmp)
        # Memory efficiency
        del tmp

        if strategy == 'binary':
            # Defines the index relative to the normal beats
            self.n_idx = self.lbe.transform(self.abs.transform(['N']))[0]
            # Defines the binary issue
            self.l_t = np_utils.to_categorical(self.l_t, num_classes=self.num_classes)
            self.l_t = self.l_t[:,self.n_idx].astype('int')
            self.l_e = np_utils.to_categorical(self.l_e, num_classes=self.num_classes)
            self.l_e = self.l_e[:,self.n_idx].astype('int')
            self.l_v = np_utils.to_categorical(self.l_v, num_classes=self.num_classes)
            self.l_v = self.l_v[:,self.n_idx].astype('int')

        # Load the data
        with h5py.File(self.inp, 'r') as dtb:
            tmp = np.hstack((dtb['t_fea'].value, dtb['t_fft'].value))
            self.train = tmp[self.m_t]
            tmp = np.hstack((dtb['e_fea'].value, dtb['e_fft'].value))
            self.valid = tmp[self.m_e]
            tmp = np.hstack((dtb['v_fea'].value, dtb['v_fft'].value))
            self.evals = tmp[self.m_v]

        # Defines the different folds on which to apply the Hyperband
        self.folds = KFold(n_splits=5, shuffle=True)

    # Application of the ML models
    # nme refers to the type of model to use
    # marker allows specific learning instance
    # max_iter refers to the amount of iterations with the hyperband algorithm
    def learn(self, nme, marker='None', max_iter=100):

        # Defines the data representation folder
        val = dict()
        val['x_train'] = self.train
        val['y_train'] = self.l_t
        val['w_train'] = sample_weight(self.l_t)
        val['x_valid'] = self.valid
        val['y_valid'] = self.l_e
        val['w_valid'] = sample_weight(self.l_e)
        # Defines the random search through cross-validation
        hyp = Hyperband(get_params, try_params, max_iter=max_iter, n_jobs=self.threads)
        res = hyp.run(nme, val, self.strategy, skip_last=1)
        res = sorted(res, key = lambda x: x[key])[0]
        # Extract the best estimator
        if nme == 'RFS':
            mod = RandomForestClassifier(**res['params'])
        if nme == 'GBT':
            mod = GradientBoostingClassifier(**res['params'])
        if nme == 'LGB':
            mod = lgb.LGBMClassifier(objective=self.strategy, **res['params'])
        if nme == 'ETS':
            mod = ExtraTreesClassifier(**res['params'])
        if nme == 'XGB':
            mod = xgb.XGBClassifier(**res['params'])
        if nme == 'SGD':
            mod = SGDClassifier(**res['params'])
        # Refit the best model
        mod.fit(val['x_train'], val['y_train'], sample_weight=val['w_train'])
        # Serialize the best obtained model
        if marker == 'None': pth = '../Results/{}_{}.pk'.format(nme, self.strategy)
        else: pth = '../Results/{}_{}_{}.pk'.format(nme, marker, self.strategy)
        joblib.dump(mod, pth)

# Defines the multi-channel networks

class DL_Model:

    # Initialization
    # input_db refers to where to gather the data
    # channels refers to a dict made for configuration
    # marker refers to the model ID
    def __init__(self, input_db, channels, marker=None):

        self.pth = input_db
        self.cls = channels
        # Constructors
        self.inp = []
        self.mrg = []
        # Output definition
        if marker: self.out = './models/MOD_{}.weights'.format(marker)
        else: self.out = './models/MOD.weights'.format(marker)
        # Handling labels
        with h5py.File(self.pth, 'r') as dtb:
            self.l_t = dtb['lab_t'].value.ravel()
            self.l_e = dtb['lab_e'].value.ravel()
            self.n_c = len(np.unique(self.l_t))

    # Defines a generator (training and testing)
    # fmt refers to whether apply it for training or testing
    # batch refers to the batch size
    def data_gen(self, fmt, batch=16):
        
        ind = 0

        while True :
            
            if fmt == 't': ann = self.l_t
            if fmt == 'e': ann = self.l_e
            # Reinitialize when going too far
            if ind + batch >= len(ann) : ind = 0
            # Initialization of data vector
            vec = []

            if self.cls['with_acc']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 3, shp[1]))
                    for idx, key in zip(range(3), ['x', 'y', 'z']):
                        ann = 'acc_{}_{}'.format(key, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            if self.cls['with_eeg']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['eeg_1_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_2_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_3_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_4_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_por']:

                 with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_nrm']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fft']:

                with h5py.File(self.pth, 'r') as dtb:
                    lst = sorted([ele for ele in dtb.keys() if ele[:3] == 'fft' and ele[-1] == fmt])
                    for key in lst: vec.append(dtb[key][ind:ind+batch])
                    del lst

            if self.cls['with_fea']:

                with h5py.File(self.pth, 'r') as dtb:
                    pca = dtb['pca_{}'.format(fmt)][ind:ind+batch]
                    fea = dtb['fea_{}'.format(fmt)][ind:ind+batch]
                    vec.append(np.hstack((pca, fea)))

            with h5py.File(self.pth, 'r') as dtb:

                lab = dtb['lab_{}'.format(fmt)][ind:ind+batch]
                # res = shuffle(lab, *vec)
                lab = np_utils.to_categorical(lab, num_classes=self.n_c)
                yield(vec, lab)
                del lab, vec #,res

            ind += batch

    # Adds a 2D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV2D(self, inp, callback):

        # Build model
        shp = (1, inp._keras_shape[1], inp._keras_shape[2])
        mod = Reshape(shp)(inp)
        mod = Convolution2D(64, (inp._keras_shape[1], 120), data_format='channels_first', kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first', kernel_initializer='he_normal')(mod)
        mod = Convolution2D(256, (1, 20))(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalAveragePooling2D()(mod)
        # Rework through dense network
        mod = Dense(mod._keras_shape[1], kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add layers to the model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a 1D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV1D(self, inp, callback):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(64, 240, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 60, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(256, 15, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalMaxPooling1D()(mod)
        # Rework through dense network
        mod = Dense(mod._keras_shape[1], kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)

        # Add model to main model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a 1D-LSTM Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_LSTM1D(self, inp, callback):

        # Defines the LSTM layer
        mod = Reshape((5, inp._keras_shape[1] // 5))(inp)
        arg = {'return_sequences': True}
        mod = LSTM(512, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': False}
        mod = LSTM(512, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        # Rework through dense network
        mod = Dense(mod._keras_shape[1], kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)

        # Add model to main model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Aims at representing both short and long patterns
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_DUALCV(self, inp, callback):

        # Build the channel for small patterns
        s_mod = Reshape((inp._keras_shape[1], 1))(inp)
        s_mod = Conv1D(64, 40, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = Conv1D(128, 20, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = Conv1D(256, 10, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = GlobalMaxPooling1D()(s_mod)

        # Build the channel for longer patterns
        l_mod = Reshape((inp._keras_shape[1], 1))(inp)
        l_mod = Conv1D(64, 360, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = Conv1D(128, 60, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = Conv1D(256, 10, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = GlobalMaxPooling1D()(l_mod)

        # Rework through dense network
        mod = concatenate([s_mod, l_mod])
        mod = Dense(mod._keras_shape[1], kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)

        # Add model to main model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a locally connected dense channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_LDENSE(self, inp, callback):

        # Build the model
        mod = Dense(inp._keras_shape[1], kernel_initializer='he_normal')(inp)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, kernel_initializer='he_normal')(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add layers to model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Defines the whole architecture
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    def build(self, dropout, decrease, n_tail):

        # Defines the dropout callback
        self.drp = DecreaseDropout(dropout, decrease)

        if self.cls['with_acc']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
                self.add_CONV2D(inp, self.drp)

        if self.cls['with_eeg']:
            with h5py.File(self.pth, 'r') as dtb:
                for key in ['eeg_1_t', 'eeg_2_t', 'eeg_3_t', 'eeg_4_t']:
                    inp = Input(shape=(dtb[key].shape[1], ))
                    self.add_LSTM1D(inp, self.drp)
                    self.add_CONV1D(inp, self.drp)

        if self.cls['with_por']:
            with h5py.File(self.pth, 'r') as dtb:
                for key in ['po_r_t', 'po_ir_t']:
                    inp = Input(shape=(dtb[key].shape[1], ))
                    self.add_LSTM1D(inp, self.drp)
                    self.add_CONV1D(inp, self.drp)

        if self.cls['with_nrm']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['norm_t'].shape[1], ))
                self.add_LSTM1D(inp, self.drp)
                self.add_CONV1D(inp, self.drp)

        if self.cls['with_fft']:
            with h5py.File(self.pth, 'r') as dtb:
                lst = sorted([ele for ele in dtb.keys() if ele[:3] == 'fft' and ele[-1] == 't'])
                for key in lst:
                    inp = Input(shape=(dtb[key].shape[1],))
                    self.add_LDENSE(inp, self.drp)

        if self.cls['with_fea']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['pca_t'].shape[1] + dtb['fea_t'].shape[1], ))
                self.add_LDENSE(inp, self.drp)

        # Gather all the model in one dense network
        print('# Ns Channels: ', len(self.mrg))
        model = concatenate(self.mrg)
        print('# Merge Layer: ', model._keras_shape[1])

        # Defines the learning tail
        tails = np.linspace(2*self.n_c, model._keras_shape[1], num=n_tail)

        for idx in range(n_tail):
            model = Dense(int(tails[n_tail - 1 - idx]), kernel_initializer='he_normal')(model)
            model = BatchNormalization()(model)
            model = PReLU()(model)
            model = AdaptiveDropout(self.drp.prb, self.drp)(model)

        # Last layer for probabilities
        model = Dense(self.n_c, activation='softmax', kernel_initializer='he_normal')(model)

        return model

    # Launch the learning process (GPU-oriented)
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    # patience is the parameter of the EarlyStopping callback
    # max_epochs refers to the amount of epochs achievable
    # batch refers to the batch_size
    def learn(self, dropout=0.5, decrease=50, n_tail=5, patience=3, max_epochs=100, batch=16):

        # Compile the model
        with tf.device('/cpu:0'): model = self.build(dropout, decrease, n_tail)
        model = Model(inputs=self.inp, outputs=model)
        arg = {'loss': 'categorical_crossentropy', 'optimizer': 'adadelta'}
        model.compile(metrics=['accuracy'], **arg)
        print('# Model Compiled')
        
        # Implements the callbacks
        arg = {'patience': patience, 'verbose': 0}
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, **arg)
        arg = {'save_best_only': True, 'save_weights_only': True}
        check = ModelCheckpoint(self.out, monitor='val_acc', **arg)
        
        # Fit the model
        model.fit_generator(self.data_gen('t', batch=32),
                            steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                            epochs=max_epochs, callbacks=[self.drp, early, check],
                            shuffle=True, validation_steps=len(self.l_e)//batch,
                            validation_data=self.data_gen('e', batch=32), 
                            class_weight=class_weight(self.l_t))
