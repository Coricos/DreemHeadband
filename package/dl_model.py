# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from package.callback import *
    
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
    def data_gen(self, fmt, batch=64):
        
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
                    tmp = np.empty((min(poi, batch), 3, shp[1]))
                    for idx, key in zip(range(3), ['x', 'y', 'z']):
                        ann = 'acc_{}_{}'.format(key, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            if self.cls['with_acc_cv1'] or self.cls['with_acc_ls1'] or self.cls['with_acc_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['acc_x', 'acc_y', 'acc_z']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_n_a_cv1'] or self.cls['with_n_a_ls1']
            if boo or self.cls['with_n_a_cvl'] or self.cls['with_n_a_dlc']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_acc_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_eeg_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['eeg_1_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 4, shp[1]))
                    for idx in range(4):
                        ann = 'eeg_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            boo = self.cls['with_eeg_cv1'] or self.cls['with_eeg_ls1']
            boo = boo or self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']
            if boo or self.cls['with_eeg_dlc'] or self.cls['with_eeg_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_tda']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['bup_1', 'bdw_1', 'bup_2', 'bdw_2', 'bup_3', 'bdw_3', 'bup_4', 'bdw_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_n_e_cv1'] or self.cls['with_n_e_ls1']
            if boo or self.cls['with_n_e_cvl'] or self.cls['with_n_e_dlc']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_eeg_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_wav_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['wav_1_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 4, shp[1]))
                    for idx in range(4):
                        ann = 'wav_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            boo = self.cls['with_wav_cv1'] or self.cls['with_wav_ls1']
            if boo or self.cls['with_wav_dlc'] or self.cls['with_wav_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['wav_1', 'wav_2', 'wav_3', 'wav_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_oxy_cv1'] or self.cls['with_oxy_ls1']
            if boo or self.cls['with_oxy_cvl'] or self.cls['with_oxy_dlc']:

                 with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fft']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fft_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fea']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_pca']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['pca_{}'.format(fmt)][ind:ind+batch])

            with h5py.File(self.pth, 'r') as dtb:

                # Defines the labels
                lab = dtb['lab_{}'.format(fmt)][ind:ind+batch]
                lab = np_utils.to_categorical(lab, num_classes=self.n_c)
                if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']:
                    lab = [lab]
                    for i in range(1, 5): lab.append(dtb['eeg_{}_{}'.format(i, fmt)][ind:ind+batch])
                yield(vec, lab)
                del lab, vec

            ind += batch

    # Defines a generator (testing and validation)
    # fmt refers to whether apply it for testing or validation
    # batch refers to the batch size
    def data_val(self, fmt, batch=512):

        if fmt == 'e': 
            sze = len(self.l_e)
        if fmt == 'v': 
            with h5py.File(self.pth, 'r') as dtb: sze = dtb['eeg_1_t'].shape[0]

        ind, poi = 0, sze

        while True :
            
            # Reinitialize when going too far
            if ind > sze : ind, poi = 0, sze
            # Initialization of data vector
            vec = []


            if self.cls['with_acc_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 3, shp[1]))
                    for idx, key in zip(range(3), ['x', 'y', 'z']):
                        ann = 'acc_{}_{}'.format(key, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            if self.cls['with_acc_cv1'] or self.cls['with_acc_ls1'] or self.cls['with_acc_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['acc_x', 'acc_y', 'acc_z']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_n_a_cv1'] or self.cls['with_n_a_ls1']
            if boo or self.cls['with_n_a_cvl'] or self.cls['with_n_a_dlc']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_acc_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_eeg_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['eeg_1_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 4, shp[1]))
                    for idx in range(4):
                        ann = 'eeg_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            boo = self.cls['with_eeg_cv1'] or self.cls['with_eeg_ls1']
            boo = boo or self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']
            if boo or self.cls['with_eeg_dlc'] or self.cls['with_eeg_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_tda']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['bup_1', 'bdw_1', 'bup_2', 'bdw_2', 'bup_3', 'bdw_3', 'bup_4', 'bdw_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_n_e_cv1'] or self.cls['with_n_e_ls1']
            if boo or self.cls['with_n_e_cvl'] or self.cls['with_n_e_dlc']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_eeg_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_wav_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['wav_1_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 4, shp[1]))
                    for idx in range(4):
                        ann = 'wav_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            boo = self.cls['with_wav_cv1'] or self.cls['with_wav_ls1']
            if boo or self.cls['with_wav_dlc'] or self.cls['with_wav_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['wav_1', 'wav_2', 'wav_3', 'wav_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            boo = self.cls['with_oxy_cv1'] or self.cls['with_oxy_ls1']
            if boo or self.cls['with_oxy_cvl'] or self.cls['with_oxy_dlc']:

                 with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fft']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fft_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fea']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_pca']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['pca_{}'.format(fmt)][ind:ind+batch])

            with h5py.File(self.pth, 'r') as dtb: yield(vec)

            ind += batch
            poi -= batch

    # Adds a 2D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_CONV2D(self, inp, callback, arg):

        # Build model
        shp = (1, inp._keras_shape[1], inp._keras_shape[2])
        mod = Reshape(shp)(inp)
        mod = Convolution2D(64, (shp[1], 210), data_format='channels_first', **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(128, (1, 8), data_format='channels_first', **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(128, (1, 8), data_format='channels_first', **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(128, (1, 8), data_format='channels_first', **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling2D(pool_size=(1, 8), strides=(1, 4), data_format='channels_first', **arg)(mod)
        mod = GlobalAveragePooling2D()(mod)

        # Add layers to the model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # 1D CNN channel designed for the TDA betti curves
    # inp refers to the defined input
    # callback refers to the annealing dropout
    # arg refers to arguments for layer initalization
    def add_TDAC1(self, inp, callback, arg):

        # Build model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(64, 10, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 4, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 4, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 4, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=4, strides=4)(mod)
        mod = GlobalAveragePooling1D()(mod)

        # Add layers to the model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a 1D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_CONV1D(self, inp, callback, arg):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(64, 210, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AveragePooling1D(pool_size=4, strides=4)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=8, strides=8)(mod)
        mod = GlobalAveragePooling1D()(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds an autoencoder channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_ENCODE(self, inp, callback, topology, arg):

        if topology == 'dense':

            # Build the autoencoder model
            enc0 = Dense(inp._keras_shape[1] // 3, **arg)(inp)
            enc = BatchNormalization()(enc0)
            enc = PReLU()(enc)
            enc = AdaptiveDropout(callback.prb, callback)(enc)
            enc1 = Dense(enc._keras_shape[1] // 4, **arg)(enc)
            enc = BatchNormalization()(enc1)
            enc = PReLU()(enc)
            enc = AdaptiveDropout(callback.prb, callback)(enc)
            enc2 = Dense(enc._keras_shape[1] // 3, **arg)(enc)
            enc = BatchNormalization()(enc2)
            enc = PReLU()(enc)
            enc = AdaptiveDropout(callback.prb, callback)(enc)
            enc3 = Dense(enc._keras_shape[1] // 4, **arg)(enc)
            enc = BatchNormalization()(enc3)
            enc = PReLU()(enc)
            enc = AdaptiveDropout(callback.prb, callback)(enc)

            print('# Latent Space Dimension', enc._keras_shape[1])

            dec = Dense(enc2._keras_shape[1], **arg)(enc)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = AdaptiveDropout(callback.prb, callback)(dec)
            dec = Dense(enc1._keras_shape[1], **arg)(dec)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = AdaptiveDropout(callback.prb, callback)(dec)
            dec = Dense(enc0._keras_shape[1], **arg)(dec)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = AdaptiveDropout(callback.prb, callback)(dec)
            arg = {'activation': 'linear', 'name': 'ate_{}'.format(len(self.ate))}
            dec = Dense(inp._keras_shape[1], kernel_initializer='he_uniform', **arg)(dec)

        if topology == 'convolution':

            # Build the autoencoder model
            enc = Reshape((inp._keras_shape[1], 1))(inp)
            enc = Conv1D(32, 70, border_mode='same', **arg)(enc)
            enc = BatchNormalization()(enc)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)
            enc = MaxPooling1D(pool_size=3)(enc)
            enc = Conv1D(64, 8, border_mode='same', **arg)(enc)
            enc = BatchNormalization()(enc)
            enc = PReLU()(enc)
            enc = Dropout(0.1)(enc)
            enc = MaxPooling1D(pool_size=3)(enc)

            dec = Conv1D(64, 8, border_mode='same', **arg)(enc)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = Dropout(0.1)(dec)
            dec = UpSampling1D(size=3)(dec)
            dec = Conv1D(32, 70, border_mode='same', **arg)(dec)
            dec = BatchNormalization()(dec)
            dec = PReLU()(dec)
            dec = Dropout(0.1)(dec)
            dec = UpSampling1D(size=3)(dec)
            dec = Conv1D(1, 70, border_mode='same', **arg)(dec)
            dec = BatchNormalization()(dec)
            dec = Activation('linear')(dec)
            arg = {'name': 'ate_{}'.format(len(self.ate))}
            dec = Reshape((dec._keras_shape[1],), **arg)(dec)

            # Returns the right encoder
            enc = GlobalAveragePooling1D()(enc)
            print('# Latent Space Dimension', enc._keras_shape)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.ate.append(dec)
        self.mrg.append(enc)

    # Adds a 1D-LSTM Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_LSTM1D(self, inp, callback, arg):

        # Defines the LSTM layer
        mod = Reshape((3, inp._keras_shape[1] // 3))(inp)
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=False, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Defines a channel combining CONV and LSTM structures
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_CVLSTM(self, inp, callback, arg):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(64, 210, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 8, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=8, strides=4)(mod)
        # Output into LSTM network
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(128, return_sequences=False, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Aims at representing both short and long patterns
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_DUALCV(self, inp, callback, arg):

        # Build the channel for small patterns
        s_mod = Reshape((inp._keras_shape[1], 1))(inp)
        s_mod = Conv1D(64, 70, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = AdaptiveDropout(callback.prb, callback)(s_mod)
        s_mod = AveragePooling1D(pool_size=8, strides=4)(s_mod)
        s_mod = GlobalAveragePooling1D()(s_mod)

        # Build the channel for longer patterns
        l_mod = Reshape((inp._keras_shape[1], 1))(inp)
        l_mod = Conv1D(64, 210, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = Conv1D(128, 8, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = Conv1D(128, 8, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = Conv1D(128, 8, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = AdaptiveDropout(callback.prb, callback)(l_mod)
        l_mod = AveragePooling1D(pool_size=8, strides=4)(l_mod)
        l_mod = GlobalAveragePooling1D()(l_mod)

        # Rework through dense network
        mod = concatenate([s_mod, l_mod])

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a locally connected dense channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_LDENSE(self, inp, callback, arg):

        # Build the model
        mod = Dense(inp._keras_shape[1] // 2, **arg)(inp)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(20, **arg)(inp)
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
        # Layer arguments
        arg = {'kernel_initializer': 'he_uniform', 
               # 'kernel_constraint': max_norm(5.0),
               # 'kernel_regularizer': regularizers.l2(1e-3)
              }

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_acc_cv2']:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
                self.add_CONV2D(inp, self.drp, arg)
            for key in ['acc_x_t', 'acc_y_t', 'acc_z_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_acc_cv1']: self.add_CONV1D(inp, self.drp, arg)
                if self.cls['with_acc_ls1']: self.add_LSTM1D(inp, self.drp, arg)
                if self.cls['with_acc_cvl']: self.add_CVLSTM(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['norm_acc_t'].shape[1], ))
            if self.cls['with_n_a_cv1']: self.add_LSTM1D(inp, self.drp, arg)
            if self.cls['with_n_a_ls1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_n_a_cvl']: self.add_CVLSTM(inp, self.drp, arg)
            if self.cls['with_n_a_dlc']: self.add_DUALCV(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_cv2']:
                inp = Input(shape=(4, dtb['eeg_1_t'].shape[1]))
                self.add_CONV2D(inp, self.drp, arg)
            for key in ['eeg_1_t', 'eeg_2_t', 'eeg_3_t', 'eeg_4_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_eeg_cv1']: self.add_CONV1D(inp, self.drp, arg)
                if self.cls['with_eeg_ls1']: self.add_LSTM1D(inp, self.drp, arg)
                if self.cls['with_eeg_dlc']: self.add_DUALCV(inp, self.drp, arg)
                if self.cls['with_eeg_cvl']: self.add_CVLSTM(inp, self.drp, arg)
                if self.cls['with_eeg_atd']: self.add_ENCODE(inp, self.drp, 'dense', arg)
                if self.cls['with_eeg_atc']: self.add_ENCODE(inp, self.drp, 'convolution', arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_tda']:
                for key in ['bup_1_t', 'bdw_1_t', 'bup_2_t', 'bdw_2_t', 'bup_3_t', 'bdw_3_t', 'bup_4_t', 'bdw_4_t']:
                    inp = Input(shape=(dtb[key].shape[1], ))
                    self.add_TDAC1(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['norm_eeg_t'].shape[1], ))
            if self.cls['with_n_e_cv1']: self.add_LSTM1D(inp, self.drp, arg)
            if self.cls['with_n_e_ls1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_n_e_cvl']: self.add_CVLSTM(inp, self.drp, arg)
            if self.cls['with_n_e_dlc']: self.add_DUALCV(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_wav_cv2']:
                inp = Input(shape=(4, dtb['wav_1_t'].shape[1]))
                self.add_CONV2D(inp, self.drp, arg)
            for key in ['wav_1_t', 'wav_2_t', 'wav_3_t', 'wav_4_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_wav_cv1']: self.add_CONV1D(inp, self.drp, arg)
                if self.cls['with_wav_ls1']: self.add_LSTM1D(inp, self.drp, arg)
                if self.cls['with_wav_dlc']: self.add_DUALCV(inp, self.drp, arg)
                if self.cls['with_wav_cvl']: self.add_CVLSTM(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            for key in ['po_r_t', 'po_ir_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_oxy_cv1']: self.add_CONV1D(inp, self.drp, arg)
                if self.cls['with_oxy_ls1']: self.add_LSTM1D(inp, self.drp, arg)
                if self.cls['with_oxy_cvl']: self.add_CVLSTM(inp, self.drp, arg)
                if self.cls['with_oxy_dlc']: self.add_DUALCV(inp, self.drp, arg)

        if self.cls['with_fft']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['fft_t'].shape[1], ))
                self.add_LDENSE(inp, self.drp, arg)

        if self.cls['with_fea']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['fea_t'].shape[1], ))
                self.add_LDENSE(inp, self.drp, arg)

        if self.cls['with_pca']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['pca_t'].shape[1], ))
                self.add_LDENSE(inp, self.drp, arg)

        # Gather all the model in one dense network
        print('# Ns Channels: ', len(self.mrg))
        model = concatenate(self.mrg)
        print('# Merge Layer: ', model._keras_shape[1])

        # Defines the learning tail
        tails = np.linspace(2*self.n_c, model._keras_shape[1], num=n_tail)

        for idx in range(n_tail):
            # Build intermediate layers
            model = Dense(int(tails[n_tail - 1 - idx]), **arg)(model)
            model = BatchNormalization()(model)
            model = PReLU()(model)
            model = AdaptiveDropout(self.drp.prb, self.drp)(model)

        # Last layer for probabilities
        arg = {'activation': 'softmax', 'name': 'output'}
        model = Dense(self.n_c, kernel_initializer='he_uniform', **arg)(model)

        return model

    # Launch the learning process (GPU-oriented)
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    # patience is the parameter of the EarlyStopping callback
    # max_epochs refers to the amount of epochs achievable
    # batch refers to the batch_size
    def learn(self, dropout=0.5, decrease=50, n_tail=8, patience=3, max_epochs=100, batch=64):

        # Compile the model
        model = self.build(dropout, decrease, n_tail)

        # Defines the losses depending on the case
        if self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']: 
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
        model = Model(inputs=self.inp, outputs=model)
        opt = Adadelta(clipnorm=1.0)
        arg = {'loss': loss, 'optimizer': opt}
        model.compile(metrics=metrics, loss_weights=loss_weights, **arg)
        print('# Model Compiled')
        
        # Fit the model
        his = model.fit_generator(self.data_gen('t', batch=batch),
                    steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                    epochs=max_epochs, callbacks=[self.drp, early, check],
                    shuffle=True, validation_steps=len(self.l_e)//batch,
                    validation_data=self.data_gen('e', batch=batch), 
                    class_weight=class_weight(self.l_t))

        # Serialize its training history
        with open(self.his, 'wb') as raw: pickle.dump(his.history, raw)

    # Generates figure of training history
    def generate_figure(self):

        # Load model history
        with open(self.his, 'rb') as raw: dic = pickle.load(raw)

        # Generates the plot
        if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']: 
            plt.figure(figsize=(18,8))
        else:
            plt.figure(figsize=(18,4))
        
        fig = gd.GridSpec(2,2)

        if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']: 
            plt.subplot(fig[0,0])
            acc, val = dic['output_acc'], dic['val_output_acc']
        else:
            plt.subplot(fig[:,0])
            acc, val = dic['acc'], dic['val_acc']

        plt.title('Accuracy Evolution - Classification')
        plt.plot(range(len(acc)), acc, c='orange', label='Train')
        plt.scatter(range(len(val)), val, marker='x', s=50, c='grey', label='Test')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']: 
            plt.subplot(fig[0,1])
            plt.title('Score Evolution - AutoEncoder')
            color_t, color_e = ['orange', 'lightblue', 'red', 'grey'], ['blue', 'yellow', 'green', 'black']
            for ate, col_t, col_e in zip(['ate_0', 'ate_1', 'ate_2', 'ate_3'], color_t, color_e):
                plt.plot(dic['{}_mean_absolute_error'.format(ate)], c=col_t, label='Train_{}'.format(ate))
                tmp = dic['val_{}_mean_absolute_error'.format(ate)]
                plt.scatter(range(len(tmp)), tmp, marker='x', s=4, c=col_e, label='Test_{}'.format(ate))
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('MAE')

        if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']: 
            plt.subplot(fig[1,:])
        else: 
            plt.subplot(fig[:,1])

        plt.title('Losses Evolution')
        plt.plot(dic['loss'], c='orange', label='Train Loss')
        plt.plot(dic['val_loss'], c='grey', label='Test Loss')

        if self.cls['with_eeg_atd'] or self.cls['with_eeg_atc']: 
            plt.plot(dic['output_loss'], c='red', label='Train Classification Loss')
            plt.plot(dic['val_output_loss'], c='darkblue', label='Test Classification Loss')
            color_t, color_e = ['orange', 'lightblue', 'red', 'grey'], ['blue', 'yellow', 'green', 'black']
            for ate, col_t, col_e in zip(['ate_0', 'ate_1', 'ate_2', 'ate_3'], color_t, color_e):
                plt.plot(dic['{}_loss'.format(ate)], c=col_t, label='Loss_Train_{}'.format(ate))
                plt.plot(dic['val_{}_loss'.format(ate)], c=col_e, label='Loss_Test_{}'.format(ate))

        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()

    # Rebuild the model from the saved weights
    # n_tail refers to the amount of layers to merge the channels
    def reconstruct(self, n_tail=8):

        # Reinitialize attributes
        self.inp, self.mrg = [], []
        
        # Build the model
        model = self.build(0.0, 100, n_tail)

        # Defines the losses depending on the case
        if self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']: 
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

        # Build and compile the model
        model = Model(inputs=self.inp, outputs=model)

        # Load the appropriate weights
        model.load_weights(self.out)
        self.clf = model
        del model

    # Validate on the unseen samples
    # fmt refers to whether apply it for testing or validation
    # n_tail refers to the marker for the model reconstruction
    # batch refers to the batch size
    def predict(self, fmt, n_tail=8, batch=512):

        # Load the best model saved
        if not hasattr(self, 'clf'): self.reconstruct(n_tail=n_tail)

        # Defines the size of the validation set
        if fmt == 'e': sze = len(self.l_e)
        if fmt == 'v': 
            with h5py.File(self.pth, 'r') as dtb: sze = dtb['eeg_1_t'].shape[0]

        # Defines the tools for prediction
        gen, ind, prd = self.data_val(fmt, batch=batch), 0, []

        for vec in gen:
            # Defines the right stop according to the batch_size
            if (sze / batch) - int(sze / batch) == 0 : end = int(sze / batch) - 1
            else : end = int(sze / batch)
            # Iterate according to the right stopping point
            if ind <= end :
                if self.cls['with_eeg_atc'] or self.cls['with_eeg_atd']:
                    prd += [np.argmax(pbs) for pbs in self.clf.predict(vec)[0]]
                else:
                    prd += [np.argmax(pbs) for pbs in self.clf.predict(vec)]
                ind += 1
            else : 
                break

        return np.asarray(prd)

    # Generates the confusion matrixes for train, test and validation sets
    # n_tail refers to the amount of layers to merge the channels
    def confusion_matrix(self, n_tail=8):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

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
        prd = self.predict('e', n_tail=n_tail)
        build_matrix(prd, self.l_e, 'TEST')
        del prd

    # Write validation to file
    # out refers to the output path
    # n_tail refers to the amount of layers to merge the channels
    def write_to_file(self, out=None, n_tail=8):

        # Compute the predictions for validation
        prd = self.predict('v', n_tail=n_tail)
        idx = np.arange(43830, 64422)
        res = np.hstack((idx.reshape(-1,1), prd.reshape(-1,1)))

        # Creates the relative dataframe
        res = pd.DataFrame(res, columns=['id', 'label'])

        # Write to csv
        if out is None: out = './results/test_{}.csv'.format(int(time.time()))
        res.to_csv(out, index=False, header=True, sep=';')
