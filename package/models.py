# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *

class DL_Model:

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

    def add_CONV2D(self, inp, dropout):

        # Build model
        mod = Reshape((1, inp._keras_shape[1], inp._keras_shape[2]))(inp)
        mod = Convolution2D(64, (inp._keras_shape[1], 60), data_format='channels_first')(mod)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(dropout)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first')(mod)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(dropout)(mod)
        mod = GlobalAveragePooling2D()(mod)
        mod = Dense(20, activation='relu')(mod)

        # Add layers to the model
        self.inp.append(inp)
        self.mrg.append(mod)

    def add_CONV1D(self, inp, dropout):

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(100, 60)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = Dropout(dropout)(mod)
        mod = Conv1D(150, 30)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(dropout)(mod)
        mod = Conv1D(200, 10)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(dropout)(mod)
        mod = GlobalMaxPooling1D()(mod)
        mod = Dense(20, activation='relu')(mod)

        # Add model to main model
        self.inp.append(inp)
        self.mrg.append(mod)

    def add_LDENSE(self, inp, dropout):

        # Build the model
        mod = Dense(inp._keras_shape[1])(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(mod._keras_shape[1] // 2)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(5, activation='relu')(mod)

        # Add layers to model
        self.inp.append(inp)
        self.mrg.append(mod)

    def build(self, dropout):

        if self.cls['with_acc']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
                self.add_CONV2D(inp, dropout)

        if self.cls['with_eeg']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(4, dtb['eeg_1_t'].shape[1]))
                self.add_CONV2D(inp, dropout)

        if self.cls['with_por']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(2, dtb['po_ir_t'].shape[1]))
                self.add_CONV2D(inp, dropout)

        if self.cls['with_nrm']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['norm_t'].shape[1], ))
                self.add_CONV1D(inp, dropout)

        if self.cls['with_fft']:
            with h5py.File(self.pth, 'r') as dtb:
                lst = sorted([ele for ele in dtb.keys() if ele[:3] == 'fft' and ele[-1] == 't'])
                for key in lst:
                    inp = Input(shape=(dtb[key].shape[1],))
                    self.add_LDENSE(inp, dropout)

        if self.cls['with_fea']:
            with h5py.File(self.pth, 'r') as dtb:
                inp = Input(shape=(dtb['pca_t'].shape[1], ))
                self.add_LDENSE(inp, dropout)

        # Gather all the model in one dense network
        model = concatenate(self.mrg)
        model = Dense(model._keras_shape[1])(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout)(model)
        model = Dense(model._keras_shape[1] // 2)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout)(model)
        model = Dense(self.n_c, activation='softmax')(model)

        return model

    def learn(self, dropout=0.33, patience=3, max_epochs=100, batch=32):

        # Compile the model
        model = self.build(dropout)
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
