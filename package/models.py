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
        self.ate = []
        self.inp = []
        self.mrg = []
        # Output definition
        if marker: self.out = './models/MOD_{}.weights'.format(marker)
        else: self.out = './models/MOD.weights'.format(marker)
        if marker: self.his = './models/HIS_{}.history'.format(marker)
        else: self.his = './models/HIS.history'
        # Handling labels
        with h5py.File(self.pth, 'r') as dtb:
            self.l_t = dtb['lab_t'].value.ravel()
            self.l_e = dtb['lab_e'].value.ravel()
            self.n_c = len(np.unique(self.l_t))

    # Defines a generator (training and testing)
    # fmt refers to whether apply it for training or testing
    # batch refers to the batch size
    def data_gen(self, fmt, batch=32):
        
        ind = 0

        while True :
            
            if fmt == 't': ann = self.l_t
            if fmt == 'e': ann = self.l_e
            # Reinitialize when going too far
            if ind + batch >= len(ann) : ind = 0
            # Initialization of data vector
            vec = []

            if self.cls['with_acc_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 3, shp[1]))
                    for idx, key in zip(range(3), ['x', 'y', 'z']):
                        ann = 'acc_{}_{}'.format(key, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            if self.cls['with_acc_cv1'] or self.cls['with_acc_ls1'] or self.cls['with_acc_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['acc_x_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['acc_y_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['acc_z_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_eeg_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['eeg_1_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 4, shp[1]))
                    for idx in range(4):
                        ann = 'eeg_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            boo = self.cls['with_eeg_cv1'] or self.cls['with_eeg_ls1']
            boo = boo or self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']
            if boo or self.cls['with_eeg_dlc'] or self.cls['with_eeg_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['eeg_1_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_2_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_3_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['eeg_4_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_oxy_atc'] or self.cls['with_oxy_atd']
            if boo or self.cls['with_oxy_cv1'] or self.cls['with_oxy_ls1'] or self.cls['with_oxy_cvl']:

                 with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_nrm_atc'] or self.cls['with_nrm_atd']
            if boo or self.cls['with_nrm_cv1'] or self.cls['with_nrm_ls1'] or self.cls['with_nrm_cvl']:

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

                # Defines the labels
                lab = dtb['lab_{}'.format(fmt)][ind:ind+batch]
                lab = np_utils.to_categorical(lab, num_classes=self.n_c)
                if self.cls['with_eeg_ate']:
                    lab = [lab]
                    for i in range(1, 5): lab.append(dtb['eeg_{}_{}'.format(i, fmt)][ind:ind+batch])
                yield(vec, lab)
                del lab, vec

            ind += batch

    # Adds a 2D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV2D(self, inp, callback):

        # Build model
        shp = (inp._keras_shape[1], 1, inp._keras_shape[2])
        mod = Reshape(shp)(inp)
        mod = Convolution2D(32, (1, 128), data_format='channels_first', kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Convolution2D(64, (1, 8), kernel_initializer='he_normal', data_format='channels_first')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalAveragePooling2D()(mod)

        # Add layers to the model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a 1D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV1D(self, inp, callback):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(16, 240, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = Conv1D(32, 8, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = Conv1D(64, 2, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalAveragePooling1D()(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds an autoencoder channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_ENCODE(self, inp, topology):

        if topology == 'dense':

            # Build the autoencoder model
            enc0 = Dense(inp._keras_shape[1] // 3, kernel_initializer='he_normal')(inp)
            enc = BatchNormalization()(enc0)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)
            enc1 = Dense(enc._keras_shape[1] // 3, kernel_initializer='he_normal')(enc)
            enc = BatchNormalization()(enc1)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)
            enc2 = Dense(enc._keras_shape[1] // 3, kernel_initializer='he_normal')(enc)
            enc = BatchNormalization()(enc2)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)
            enc3 = Dense(enc._keras_shape[1] // 3, kernel_initializer='he_normal')(enc)
            enc = BatchNormalization()(enc3)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)

            print('# Latent Space Dimension', enc._keras_shape[1])

            dec = Dense(enc2._keras_shape[1], kernel_initializer='he_normal')(enc)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = Dropout(0.1)(dec)
            dec = Dense(enc1._keras_shape[1], kernel_initializer='he_normal')(dec)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = Dropout(0.1)(dec)
            dec = Dense(enc0._keras_shape[1], kernel_initializer='he_normal')(dec)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = Dropout(0.1)(dec)
            arg = {'activation': 'linear', 'name': 'ate_{}'.format(len(self.ate))}
            dec = Dense(inp._keras_shape[1], kernel_initializer='he_normal', **arg)(dec)

        if topology == 'convolution':

            # Build the autoencoder model
            enc = Reshape((inp._keras_shape[1], 1))(inp)
            enc = Conv1D(32, 128, activation='relu', border_mode='same', kernel_initializer='he_normal')(enc)
            enc = AveragePooling1D(pool_size=5)(enc)
            enc = Conv1D(32, 8, activation='relu', border_mode='same', kernel_initializer='he_normal')(enc)
            enc = AveragePooling1D(pool_size=4)(enc)
            
            print('# Latent Space Dimension', enc._keras_shape)

            dec = Conv1D(32, 8, activation='relu', border_mode='same', kernel_initializer='he_normal')(enc)
            dec = UpSampling1D(size=4)(dec)
            dec = Conv1D(32, 128, activation='relu', border_mode='same', kernel_initializer='he_normal')(dec)
            dec = UpSampling1D(size=5)(dec)
            dec = Conv1D(1, 128, activation='linear', border_mode='same', kernel_initializer='he_normal')(dec)
            arg = {'name': 'ate_{}'.format(len(self.ate))}
            dec = Reshape((dec._keras_shape[1],), **arg)(dec)

            # Returns the right encoder
            enc = GlobalAveragePooling1D()(enc)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.ate.append(dec)
        self.mrg.append(enc)

    # Adds a 1D-LSTM Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_LSTM1D(self, inp, callback):

        # Defines the LSTM layer
        mod = Reshape((5, inp._keras_shape[1] // 5))(inp)
        arg = {'return_sequences': True}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': True}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': True}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': False}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Defines a channel combining CONV and LSTM structures
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CVLSTM(self, inp, callback):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(16, 240, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = Conv1D(64, 8, kernel_initializer='he_normal')(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        # Output into LSTM network
        arg = {'return_sequences': True}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': True}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        arg = {'return_sequences': False}
        mod = LSTM(64, kernel_initializer='he_normal', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Aims at representing both short and long patterns
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_DUALCV(self, inp, callback):

        # Build the channel for small patterns
        s_mod = Reshape((inp._keras_shape[1], 1))(inp)
        s_mod = Conv1D(16, 70, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = AveragePooling1D(pool_size=2)(s_mod)
        s_mod = Conv1D(32, 8, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = AveragePooling1D(pool_size=2)(s_mod)
        s_mod = Conv1D(64, 4, kernel_initializer='he_normal')(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = GlobalAveragePooling1D()(s_mod)

        # Build the channel for longer patterns
        l_mod = Reshape((inp._keras_shape[1], 1))(inp)
        l_mod = Conv1D(8, 210, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = AveragePooling1D(pool_size=2)(l_mod)
        l_mod = Conv1D(16, 8, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = AveragePooling1D(pool_size=2)(l_mod)
        l_mod = Conv1D(32, 4, kernel_initializer='he_normal')(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = GlobalAveragePooling1D()(l_mod)

        # Rework through dense network
        mod = concatenate([s_mod, l_mod])

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
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

        # Add layers to model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Defines the whole architecture
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    def build(self, dropout, decrease, n_tail):

        # Defines the dropout callback
        self.drp = DecreaseDropout(dropout, decrease)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_acc_cv2']:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
                self.add_CONV2D(inp, self.drp)
            for key in ['acc_x_t', 'acc_y_t', 'acc_z_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_acc_cv1']: self.add_CONV1D(inp, self.drp)
                if self.cls['with_acc_ls1']: self.add_LSTM1D(inp, self.drp)
                if self.cls['with_acc_cvl']: self.add_CVLSTM(inp, self.drp)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_cv2']:
                inp = Input(shape=(4, dtb['eeg_1_t'].shape[1]))
                self.add_CONV2D(inp, self.drp)
            for key in ['eeg_1_t', 'eeg_2_t', 'eeg_3_t', 'eeg_4_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_eeg_cv1']: self.add_CONV1D(inp, self.drp)
                if self.cls['with_eeg_ls1']: self.add_LSTM1D(inp, self.drp)
                if self.cls['with_eeg_dlc']: self.add_DUALCV(inp, self.drp)
                if self.cls['with_eeg_cvl']: self.add_CVLSTM(inp, self.drp)
                if self.cls['with_eeg_atd']: self.add_ENCODE(inp, 'dense')
                if self.cls['with_eeg_atc']: self.add_ENCODE(inp, 'convolution')                

        with h5py.File(self.pth, 'r') as dtb:
            for key in ['po_r_t', 'po_ir_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_oxy_cv1']: self.add_CONV1D(inp, self.drp)
                if self.cls['with_oxy_ls1']: self.add_LSTM1D(inp, self.drp)
                if self.cls['with_oxy_cvl']: self.add_CVLSTM(inp, self.drp)
                if self.cls['with_oxy_atd']: self.add_ENDOCE(inp, 'dense')
                if self.cls['with_oxy_atc']: self.add_ENCODE(inp, 'convolution')

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['norm_t'].shape[1], ))
            if self.cls['with_nrm_cv1']: self.add_LSTM1D(inp, self.drp)
            if self.cls['with_nrm_ls1']: self.add_CONV1D(inp, self.drp)
            if self.cls['with_nrm_cvl']: self.add_CVLSTM(inp, self.drp)
            if self.cls['with_nrm_atd']: self.add_ENDOCE(inp, 'dense')
            if self.cls['with_nrm_atc']: self.add_ENCODE(inp, 'convolution')

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
        arg = {'activation': 'softmax', 'name': 'output'}
        model = Dense(self.n_c, kernel_initializer='he_normal', **arg)(model)

        return model

    # Launch the learning process (GPU-oriented)
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    # patience is the parameter of the EarlyStopping callback
    # max_epochs refers to the amount of epochs achievable
    # batch refers to the batch_size
    def learn(self, dropout=0.5, decrease=50, n_tail=8, patience=3, max_epochs=100, batch=32):

        # Compile the model
        model = self.build(dropout, decrease, n_tail)

        # Defines the losses depending on the case
        if self.cls['with_eeg_ate']: 
            model = [model] + self.ate
            loss = {'output': 'categorical_crossentropy', 
                    'ate_0': 'mean_squared_error', 'ate_1': 'mean_squared_error',
                    'ate_2': 'mean_squared_error', 'ate_3': 'mean_squared_error'}
            loss_weights = {'output': 0.5, 'ate_0': 0.25, 'ate_1': 0.25, 
                            'ate_2': 0.25, 'ate_3': 0.25}
            metrics = {'output': 'accuracy', 'ate_0': 'mae', 'ate_1': 'mae', 
                       'ate_2': 'mae', 'ate_3': 'mae'}
        else: 
            loss = 'categorical_crossentropy'
            loss_weights = None
            metrics = ['accuracy']

        # Implements the model and its callbacks
        arg = {'patience': patience, 'verbose': 0}
        early = EarlyStopping(monitor='val_loss', min_delta=1e-5, **arg)
        arg = {'save_best_only': True, 'save_weights_only': True}
        check = ModelCheckpoint(self.out, monitor='val_loss', **arg)

        # Build and compile the model
        try: model = multi_gpu_model(Model(inputs=self.inp, outputs=model))
        except: model = Model(inputs=self.inp, outputs=model)
        arg = {'optimizer': 'adadelta'}
        model.compile(metrics=metrics, loss=loss, loss_weights=loss_weights, **arg)
        print('# Model Compiled')
        
        # Fit the model
        his = model.fit_generator(self.data_gen('t', batch=batch),
                    steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                    epochs=max_epochs, callbacks=[self.drp, early, check],
                    shuffle=True, validation_steps=len(self.l_e)//batch,
                    validation_data=self.data_gen('e', batch=batch), 
                    class_weight=class_weight(self.l_t))

        # Serialize its training history
        with open(self.his, 'wb') as raw: pickle.dump(his, raw)