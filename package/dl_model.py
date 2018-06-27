# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from package.callback import *
from package.ds_model import *
    
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
                    tmp = np.empty((batch, 3, shp[1]))
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'acc_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_acc_cv1'] or self.cls['with_acc_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['acc_x', 'acc_y', 'acc_z']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_n_a_cv1'] or self.cls['with_n_a_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_acc_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_eeg_cv2']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['eeg_1_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 4, shp[1]))
                    for idx in range(4):
                        key = 'eeg_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            boo = self.cls['with_eeg_enc'] or self.cls['with_eeg_ate']
            if self.cls['with_eeg_cv1'] or self.cls['with_eeg_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_tda']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['bup_1', 'bup_2', 'bup_3', 'bup_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_l_0']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['l_0_1', 'l_0_2', 'l_0_3', 'l_0_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_l_1']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['l_1_1', 'l_1_2', 'l_1_3', 'l_1_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_n_e_cv1'] or self.cls['with_n_e_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_eeg_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_por_enc'] or self.cls['with_por_ate']
            if self.cls['with_por_cv1'] or self.cls['with_por_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_poi_enc'] or self.cls['with_poi_ate']
            if self.cls['with_poi_cv1'] or self.cls['with_poi_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fea']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(fmt)][ind:ind+batch])

            with h5py.File(self.pth, 'r') as dtb:

                # Defines the labels
                lab = dtb['lab_{}'.format(fmt)][ind:ind+batch]
                lab = np_utils.to_categorical(lab, num_classes=self.n_c)
            
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
            with h5py.File(self.pth, 'r') as dtb: sze = dtb['eeg_1_v'].shape[0]

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
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'acc_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_acc_cv1'] or self.cls['with_acc_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['acc_x', 'acc_y', 'acc_z']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_n_a_cv1'] or self.cls['with_n_a_cvl']:

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

            boo = self.cls['with_eeg_enc'] or self.cls['with_eeg_ate']
            if self.cls['with_eeg_cv1'] or self.cls['with_eeg_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_tda']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['bup_1', 'bup_2', 'bup_3', 'bup_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_l_0']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['l_0_1', 'l_0_2', 'l_0_3', 'l_0_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_eeg_l_1']:

                with h5py.File(self.pth, 'r') as dtb:
                    for key in ['l_1_1', 'l_1_2', 'l_1_3', 'l_1_4']:
                        vec.append(dtb['{}_{}'.format(key, fmt)][ind:ind+batch])

            if self.cls['with_n_e_cv1'] or self.cls['with_n_e_cvl']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['norm_eeg_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_por_enc'] or self.cls['with_por_ate']
            if self.cls['with_por_cv1'] or self.cls['with_por_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_r_{}'.format(fmt)][ind:ind+batch])

            boo = self.cls['with_poi_enc'] or self.cls['with_poi_ate']
            if self.cls['with_poi_cv1'] or self.cls['with_poi_cvl'] or boo:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['po_ir_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fea']:

                with h5py.File(self.pth, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(fmt)][ind:ind+batch])

            yield(vec)

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
        mod = Convolution2D(64, (shp[1], 32), data_format='channels_first', **arg)(mod)
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
        mod = AveragePooling2D(pool_size=(1, 8), data_format='channels_first')(mod)
        mod = GlobalAveragePooling2D()(mod)

        # Add layers to the model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # 1D CNN channel designed for the TDA betti curves
    # inp refers to the defined input
    # callback refers to the annealing dropout
    # arg refers to arguments for layer initalization
    def add_TDACV1(self, inp, callback, arg):

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
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = GlobalAveragePooling1D()(mod)

        # Add layers to the model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # 1D CNN channel designed for the TDA betti curves
    # inp refers to the defined input
    # callback refers to the annealing dropout
    # arg refers to arguments for layer initalization
    def add_SILHOU(self, inp, callback, arg):

        # Build silhouette layer
        sil = SilhouetteLayer(int(inp._keras_shape[-1]))(inp)
        mod = Conv1D(64, 10, **arg)(sil)
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
        mod = AveragePooling1D(pool_size=2)(mod)
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
        mod = Conv1D(64, 32, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 6, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 6, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = GlobalAveragePooling1D()(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds an autoencoder channel
    # inp refers to the defined input
    # channel refers to a specific hannel
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_ENCODE(self, inp, channel, callback, arg):

        enc = AutoEncoder(channel, storage='/'.join(self.pth.split('/')[:-1]))
        enc = enc.get_encoder()
        # Make it non-trainable
        for layer in enc.layers: layer.trainable = False

        mod = enc(inp)
        mod = GlobalMaxPooling1D()(mod)
        mod = Dense(mod._keras_shape[1] // 2, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds an autoencoder channel
    # inp refers to the defined input
    # channel refers to a specific hannel
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_ATENCO(self, inp, channel, callback, arg):

        ate = AutoEncoder(channel, storage='/'.join(self.pth.split('/')[:-1]))
        ate = ate.get_autoencoder()
        # Make it non-trainable
        for layer in ate.layers: layer.trainable = False

        mod = ate(inp)
        mod = Reshape((inp._keras_shape[1], 1))(mod)
        mod = Conv1D(64, 32, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 6, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 6, **arg)(mod)
        mod = PReLU()(mod)
        mod = BatchNormalization()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = AveragePooling1D(pool_size=2)(mod)
        mod = GlobalAveragePooling1D()(mod)

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
        mod = Conv1D(64, 32, **arg)(mod)
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
        mod = AveragePooling1D(pool_size=8)(mod)
        # Output into LSTM network
        mod = LSTM(64, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(64, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(64, return_sequences=True, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = LSTM(64, return_sequences=False, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        # Add model to main model
        if inp not in self.inp: self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a locally connected dense channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    # arg refers to arguments for layer initalization
    def add_LDENSE(self, inp, callback, arg):

        # Build the model
        mod = Dense(inp._keras_shape[1], **arg)(inp)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2, **arg)(mod)
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
                if self.cls['with_acc_cvl']: self.add_CVLSTM(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['norm_acc_t'].shape[1], ))
            if self.cls['with_n_a_cv1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_n_a_cvl']: self.add_CVLSTM(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_cv2']:
                inp = Input(shape=(4, dtb['eeg_1_t'].shape[1]))
                self.add_CONV2D(inp, self.drp, arg)
            for key in ['eeg_1_t', 'eeg_2_t', 'eeg_3_t', 'eeg_4_t']:
                inp = Input(shape=(dtb[key].shape[1], ))
                if self.cls['with_eeg_cv1']: self.add_CONV1D(inp, self.drp, arg)
                if self.cls['with_eeg_cvl']: self.add_CVLSTM(inp, self.drp, arg)
                if self.cls['with_eeg_enc']: self.add_ENCODE(inp, key[:-2], self.drp, arg)
                if self.cls['with_eeg_ate']: self.add_ATENCO(inp, key[:-2], self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_tda']:
                for key in ['bup_1_t', 'bup_2_t', 'bup_3_t', 'bup_4_t']:
                    inp = Input(shape=(dtb[key].shape[1], ))
                    self.add_TDACV1(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_l_0']:
                for key in ['l_0_1_t', 'l_0_2_t', 'l_0_3_t', 'l_0_4_t']:
                    inp = Input(shape=(dtb[key].shape[1], dtb[key].shape[2]))
                    self.add_SILHOU(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            if self.cls['with_eeg_l_1']:
                for key in ['l_1_1_t', 'l_1_2_t', 'l_1_3_t', 'l_1_4_t']:
                    inp = Input(shape=(dtb[key].shape[1], dtb[key].shape[2]))
                    self.add_SILHOU(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['norm_eeg_t'].shape[1], ))
            if self.cls['with_n_e_cv1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_n_e_cvl']: self.add_CVLSTM(inp, self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['po_r_t'].shape[1], ))
            if self.cls['with_por_cv1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_por_cvl']: self.add_CVLSTM(inp, self.drp, arg)
            if self.cls['with_por_enc']: self.add_ENCODE(inp, 'po_r', self.drp, arg)
            if self.cls['with_por_ate']: self.add_ATENCO(inp, 'po_r', self.drp, arg)

        with h5py.File(self.pth, 'r') as dtb:
            inp = Input(shape=(dtb['po_ir_t'].shape[1], ))
            if self.cls['with_poi_cv1']: self.add_CONV1D(inp, self.drp, arg)
            if self.cls['with_poi_cvl']: self.add_CVLSTM(inp, self.drp, arg)
            if self.cls['with_poi_enc']: self.add_ENCODE(inp, 'po_ir', self.drp, arg)
            if self.cls['with_poi_ate']: self.add_ATENCO(inp, 'po_ir', self.drp, arg)

        if self.cls['with_fea']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['fea_t'].shape[1], ))
                self.add_LDENSE(inp, self.drp, arg)

        # Gather all the model in one dense network
        print('# Ns Channels:', len(self.mrg))
        if len(self.mrg) > 1: merge = concatenate(self.mrg)
        else: merge = self.mrg[0]
        print('# Merge Layer:', merge._keras_shape[1])

        # Defines the feature encoder part
        model = Dense(model._keras_shape[1], **arg)(merge)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(model._keras_shape[1] // 3, **arg)(model)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        enc_0 = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(model._keras_shape[1] // 3, **arg)(enc_0)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        enc_1 = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(model._keras_shape[1] // 3, **arg)(enc_1)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        enc_2 = AdaptiveDropout(self.drp.prb, self.drp)(model)
        print('# Latent Space:', enc_2._keras_shape[1])
        # Defines the decoder part
        model = Dense(enc_1._keras_shape[1], **arg)(enc_2)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(enc_0._keras_shape[1], **arg)(model)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        new = {'activation': 'linear', 'name': 'decode'}
        decod = Dense(merge._keras_shape[1], **arg, **new)(model)
        # Defines the output part
        model = Dense(enc_2._keras_shape[1], **arg)(enc_2)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        new = {'activation': 'softmax', 'name': 'output'}
        model = Dense(self.n_c, **arg, **new)(model)
       
        return decod, model

    # Launch the learning process (GPU-oriented)
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    # patience is the parameter of the EarlyStopping callback
    # max_epochs refers to the amount of epochs achievable
    # batch refers to the batch_size
    def learn(self, dropout=0.5, decrease=100, n_tail=8, patience=3, max_epochs=100, batch=64):

        # Compile the model
        decod, model = self.build(dropout, decrease, n_tail)

        # Defines the losses depending on the case
        loss = {'output': 'categorical_crossentropy', 'decode': 'mean_squared_error'}
        loss_weights = {'output': 1.0, 'decode': 2.0}
        metrics = {'output': 'accuracy', 'decode': 'mean_absolute_error'}
        monitor = 'val_output_acc'

        # Implements the model and its callbacks
        arg = {'patience': patience, 'verbose': 0}
        early = EarlyStopping(monitor=monitor, min_delta=1e-5, **arg)
        arg = {'save_best_only': True, 'save_weights_only': True}
        check = ModelCheckpoint(self.out, monitor=monitor, **arg)
        if (len(self.l_e)/512) - int(len(self.l_e)/512) == 0 : steps = int(len(self.l_e)/512) -1
        else : steps = int(len(self.l_e)/512)
        kappa = Metrics(self.data_gen('e', batch=512), steps)
        shuff = DataShuffler(self.pth, 3)

        # Build and compile the model
        model = Model(inputs=self.inp, outputs=[decod, model])
        optim = Adadelta(clipnorm=1.0)
        arg = {'loss': loss, 'optimizer': optim}
        model.compile(metrics=metrics, loss_weights=loss_weights, **arg)
        print('# Model Compiled')
        
        # Fit the model
        his = model.fit_generator(self.data_gen('t', batch=batch),
                    steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                    epochs=max_epochs, callbacks=[kappa, self.drp, early, check, shuff],
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
        plt.figure(figsize=(18,4))        
        fig = gd.GridSpec(2,2)

        plt.subplot(fig[:,0])
        acc, val = dic['acc'], dic['val_acc']
        plt.title('Accuracy Evolution - Classification')
        plt.plot(range(len(acc)), acc, c='orange', label='Train')
        plt.scatter(range(len(val)), val, marker='x', s=50, c='grey', label='Test')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.subplot(fig[:,1])
        plt.title('Losses Evolution')
        plt.plot(dic['loss'], c='orange', label='Train Loss')
        plt.plot(dic['val_loss'], c='grey', label='Test Loss')

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
            with h5py.File(self.pth, 'r') as dtb: sze = dtb['eeg_1_v'].shape[0]

        # Defines the tools for prediction
        gen, ind, prd = self.data_val(fmt, batch=batch), 0, []

        for vec in gen:
            # Defines the right stop according to the batch_size
            if (sze / batch) - int(sze / batch) == 0 : end = int(sze / batch)- 1
            else : end = int(sze / batch)
            # Iterate according to the right stopping point
            if ind <= end :
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

# Defines a structure for a cross_validation

class CV_DL_Model:

    # Initialization
    # channels refers to which channels to use during multiple tests
    # storage refers to the absolute path towards the datasets
    # n_iter refers to the amount of iterations
    def __init__(self, channels, storage='./dataset', n_iter=7):

        # Attributes
        self.path = '{}/CV_Headband.h5'.format(storage)
        self.n_iter = n_iter
        self.channels = channels
        self.storage = storage

    # CV Launcher definition
    def launch(self):

        for idx in range(self.n_iter):

            # Build the new relative database
            Database(storage=self.storage).preprocess(self.path, test=0.3)

            # Launch the model scoring for each iteration
            mod = DL_Model(self.path, self.channels, marker='CV_{}'.format(idx))
            mod.learn(patience=10, dropout=0.3, decrease=100, batch=64, n_tail=5)
            mod.write_to_file(n_tail=5)

