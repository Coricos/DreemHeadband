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
        self.inp = []
        self.mrg = []
        # Output definition
        if marker: self.out = './models/MOD_{}.ks'.format(marker)
        else: self.out = './models/MOD.ks'.format(marker)
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
                    shp = dtb['eeg_1_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 4, shp[1]))
                    for idx in range(4):
                        ann = 'eeg_{}_{}'.format(idx+1, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

            if self.cls['with_por']:

                with h5py.File(self.pth, 'r') as dtb:
                    shp = dtb['po_r_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 2, shp[1]))
                    for idx, key in zip(range(2), ['r', 'ir']):
                        ann = 'po_{}_{}'.format(key, fmt)
                        tmp[:,idx,:] = dtb[ann][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, ann

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
                    vec.append(dtb['pca_{}'.format(fmt)][ind:ind+batch])

            with h5py.File(self.pth, 'r') as dtb:

                lab = dtb['lab_{}'.format(fmt)][ind:ind+batch]
                res = shuffle(lab, *vec)
                lab = np_utils.to_categorical(res[0], num_classes=self.n_c)
                yield(res[1:], lab)
                del lab, res

            ind += batch

    # Adds a 2D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV2D(self, inp, callback):

        # Build model
        mod = Reshape((1, inp._keras_shape[1], inp._keras_shape[2]))(inp)
        mod = Convolution2D(64, (inp._keras_shape[1], 60), data_format='channels_first')(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first')(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(256, (1, 15), data_format='channels_first')(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalAveragePooling2D()(mod)
        # Rework through dense network
        mod = Dense(mod._keras_shape[1])(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = MaxoutDense(mod._keras_shape[1] // 3)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)

        # Add layers to the model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a 1D-Convolution Channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_CONV1D(self, inp, callback):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(64, 60)(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(128, 30)(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(256, 15)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = GlobalMaxPooling1D()(mod)
        # Rework through dense network
        mod = Dense(mod._keras_shape[1])(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = MaxoutDense(mod._keras_shape[1] // 3)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)

        # Add model to main model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Adds a locally connected dense channel
    # inp refers to the defined input
    # callback refers to the callback managing the dropout rate 
    def add_LDENSE(self, inp, callback):

        # Build the model
        mod = Dense(inp._keras_shape[1])(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = MaxoutDense(mod._keras_shape[1] // 2)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = MaxoutDense(mod._keras_shape[1] // 2)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)

        # Add layers to model
        self.inp.append(inp)
        self.mrg.append(mod)

    # Defines the whole architecture
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    def build(self, dropout, decrease, n_tail):

        # Defines the dropout callback
        drp = DecreaseDropout(dropout, decrease)

        if self.cls['with_acc']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
                self.add_CONV2D(inp, drp)

        if self.cls['with_eeg']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(4, dtb['eeg_1_t'].shape[1]))
                self.add_CONV2D(inp, drp)

        if self.cls['with_por']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(2, dtb['po_ir_t'].shape[1]))
                self.add_CONV2D(inp, drp)

        if self.cls['with_nrm']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['norm_t'].shape[1], ))
                self.add_CONV1D(inp, drp)

        if self.cls['with_fft']:
            with h5py.File(self.pth, 'r') as dtb:
                lst = sorted([ele for ele in dtb.keys() if ele[:3] == 'fft' and ele[-1] == 't'])
                for key in lst:
                    inp = Input(shape=(dtb[key].shape[1],))
                    self.add_LDENSE(inp, drp)

        if self.cls['with_fea']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['pca_t'].shape[1], ))
                self.add_LDENSE(inp, drp)

        # Gather all the model in one dense network
        print('# Ns Channels: ', len(self.mrg))
        model = concatenate(self.mrg)
        print('# Merge Layer: ', model._keras_shape[1])

        # Defines the learning tail
        tails = np.linspace(2*self.n_c, 1.5*model._keras_shape[1], num=n_tail)
        for idx in range(n_tail):
            model = Dense(int(tails[n_tail - 1 - idx]))(model)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = AdaptiveDropout(drp.prb, drp)(model)

        # Last layer for probabilities
        model = MaxoutDense(self.n_c, activation='softmax')(model)

        return model

    # Launch the learning process (GPU-oriented)
    # dropout refers to the initial dropout rate
    # decrease refers to the amount of epochs for full annealation
    # n_tail refers to the amount of layers in the concatenate section
    # patience is the parameter of the EarlyStopping callback
    # max_epochs refers to the amount of epochs achievable
    # batch refers to the batch_size
    def learn(self, dropout=0.33, decrease=100, n_tail=5, patience=3, max_epochs=100, batch=32):

        # Compile the model
        with tf.device('/cpu:0'):
            model = self.build(dropout, decrease, n_tail)
            model = Model(inputs=self.inp, outputs=model)
            arg = {'loss': 'categorical_crossentropy', 'optimizer': 'adadelta'}
            model.compile(metrics=['accuracy'], **arg)
        
        # Implements the callbacks
        arg = {'patience': patience, 'verbose': 0}
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, **arg)
        arg = {'save_best_only': True, 'save_weights_only': False}
        check = ModelCheckpoint(self.out, monitor='val_acc', **arg)
        
        # Fit the model
        model.fit_generator(self.data_gen('t', batch=32),
                            steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                            epochs=max_epochs, callbacks=[early, check],
                            shuffle=True, validation_steps=len(self.l_e) // batch,
                            validation_data=self.data_gen('e', batch=32), 
                            class_weight=class_weight(self.l_t))
