# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from package.callback import *

# Defines the autoencoder model

class AutoEncoder:

    # Initialization
    # channel refers to the type of signal for which to apply the convolutional autoencoder
    # storage refers to where to fetch the data
    def __init__(self, channel, storage='./dataset'):

        self.dtb = storage
        # Path for checkpoint 
        self.ate = './models/ATE_{}.weights'.format(channel)
        self.his = './models/HIS_ATE_{}.history'.format(channel)
        self.enc = './models/ENC_{}.weights'.format(channel)

        with h5py.File('{}/sca_train.h5'.format(storage), 'r') as dtb:
            self.raw_t = dtb[channel].value
            self.lab_t = dtb['lab'].value.ravel()
        with h5py.File('{}/sca_valid.h5'.format(storage), 'r') as dtb:
            self.raw_v = dtb[channel].value

    # Defines a bootstraping technique for dataset augmentation
    def bootstrap(self):

        # Estimate the amount of samples to generate from each sample
        def ratios(lab, factor=2):
    
            dic = dict()
            
            for ele in np.unique(lab): dic[ele] = len(np.where(lab == ele)[0])
            m_x = max(list(dic.values()))
            for ele in dic.keys(): dic[ele] = int(factor * m_x / dic[ele])
            
            return dic

        dic, val = ratios(self.lab_t), []
        # Iterates over all the different labels
        for key in np.unique(self.lab_t):
            tmp = self.raw_t[np.where(self.lab_t == key)[0]]
            fun = partial(bootstrap_sample, num=ratios(self.lab_t)[key]-1)
            pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            val.append(np.vstack(tuple(pol.map(fun, tmp))))
            pol.close()
            pol.join()

        # Shuffle and save as attribute
        self.raw = shuffle(np.vstack(tuple(val)))
        self.raw = shuffle(np.vstack((self.raw, self.raw_v)))

        # Memory efficiency
        del self.raw_t, self.lab_t, self.raw_v, dic, val

    # Build the relative model for drowsiness classification
    # dropout refers to the amount of dropout to set in the model
    # decrease refers to the number of epochs for decreasing ratio
    def build(self, dropout, decrease):

        self.inp = Input(shape=(self.raw.shape[1], ))
        self.drp = DecreaseDropout(dropout, decrease)

        arg = {'kernel_initializer': 'he_uniform'}

        mod = Reshape((self.inp._keras_shape[1], 1))(self.inp)
        mod = GaussianNoise(np.std(self.raw) / 2)(mod)
        mod = Conv1D(64, 32, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = MaxPooling1D(pool_size=5)(mod)
        mod = AdaptiveDropout(self.drp.prb, self.drp)(mod)
        mod = Conv1D(128, 6, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = MaxPooling1D(pool_size=5)(mod)
        enc = GaussianNoise(1e-3)(mod)
        print('# ENCODER Latent Space', enc._keras_shape)

        mod = Conv1D(128, 6, padding='same', **arg)(enc)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = UpSampling1D(5)(mod)
        mod = AdaptiveDropout(self.drp.prb, self.drp)(mod)
        mod = Conv1D(64, 32, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = UpSampling1D(5)(mod)
        mod = Conv1D(1, 32, padding='same', **arg)(mod)
        mod = AdaptiveDropout(self.drp.prb, self.drp)(mod)
        mod = Activation('linear')(mod)
        dec = Flatten(name='decoder')(mod) 

        return dec
    
    # Model training
    # test_ratio gives 
    # batch_size refers to the used batch_size during learning
    # dropout is the dropout rate into the model
    # patience is the parameter of the EarlyStopping callback
    # verbose indicates whether to use tensorboard or not
    # max_epochs refers to the amount of epochs achievable
    def learn(self, test_ratio=0.3, dropout=0.25, decrease=100, batch_size=64, patience=5, verbose=1, max_epochs=100):

        # Apply data augmentation
        self.bootstrap()

        # Build the model
        model = self.build(dropout, decrease)
        model = Model(inputs=self.inp, outputs=model)
        model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adadelta')

        # Defines the callbacks
        early = EarlyStopping(monitor='loss', min_delta=1e-5, patience=patience, mode='min')
        check = ModelCheckpoint(self.ate, period=1, monitor='loss', mode='min', 
                                save_best_only=True, save_weights_only=True)

        # Launch the learning
        his = model.fit(self.raw, self.raw, verbose=verbose, epochs=max_epochs, batch_size=batch_size,
                        shuffle=True, callbacks=[self.drp, check, early], validation_split=test_ratio)

        # Save model history
        with open(self.his, 'wb') as raw: pickle.dump(his.history, raw)

        # Memory efficiency
        del model, early, check, his

    # Reconstruct the model
    def get_autoencoder(self):

        # Build the model
        model = self.build(0.0, 100)
        model = Model(inputs=self.inp, outputs=model)
        model.load_weights(self.ate)

        return model

    # Random visualization over the autoencoder reconstruction power
    def see_result(self):

        # Randomly select an index among the possibilities
        idx = np.random.choice(self.raw.shape[0])
        ate = self.get_autoencoder()

        plt.figure(figsize=(18,4))
        plt.plot(self.raw[idx], label='Initial Signal')
        plt.scatter(np.arange(self.raw.shape[1]), ate.predict(self.raw[idx].reshape(1,self.raw.shape[1])), c='b', marker='x')
        plt.show()

    # Reconstruct the encoder
    def get_encoder(self):

        # Build the model
        model = self.get_autoencoder()
        model = Model(inputs=model.input, outputs=model.layers[-14].output)

        return model

# Prebuild an entire Convolutional Channel

class CV1_Channel:

    # Initialization
    # channel refers to the type of signal for which to apply the convolutional autoencoder
    # test_ratio needed for training and testing
    # storage refers to where to fetch the data
    def __init__(self, channel, test_ratio=0.25, storage='./dataset'):

        self.dtb = storage
        # Path for checkpoint 
        self.ate = './models/CV1_{}.weights'.format(channel)
        self.his = './models/HIS_CV1_{}.history'.format(channel)
        self.enc = './models/CV1_{}.weights'.format(channel)

        with h5py.File('{}/DTB_Headband.h5'.format(storage), 'r') as dtb:
            self.raw_t = dtb[channel + '_t'].value
            self.lab_t = dtb['lab_t'].value.ravel()
            self.raw_e = dtb[channel + '_e'].value
            self.lab_e = dtb['lab_e'].value.ravel()

    # Defines a bootstraping technique for dataset augmentation
    def bootstrap(self):

        # Estimate the amount of samples to generate from each sample
        def ratios(lab, factor=2):
    
            dic = dict()
            
            for ele in np.unique(lab): dic[ele] = len(np.where(lab == ele)[0])
            m_x = max(list(dic.values()))
            for ele in dic.keys(): dic[ele] = int(factor * m_x / dic[ele])
            
            return dic

        dic, val, lbl = ratios(self.lab_t), [], []
        # Iterates over all the different labels
        for key in np.unique(self.lab_t):
            tmp = self.raw_t[np.where(self.lab_t == key)[0]]
            fun = partial(bootstrap_sample, num=ratios(self.lab_t)[key]-1)
            pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            val.append(np.vstack(tuple(pol.map(fun, tmp))))
            lbl += list(np.full(len(val[-1]), key))
            pol.close()
            pol.join()

        # Shuffle and save as attribute
        self.raw_t, self.lab_t = shuffle(np.vstack(tuple(val)), np.asarray(lbl))

        # Memory efficiency
        del dic, val

    # Build the relative model for drowsiness classification
    # dropout refers to the amount of dropout to set in the model
    # decrease refers to the number of epochs for decreasing ratio
    def build(self, dropout, decrease):

        self.inp = Input(shape=(self.raw_t.shape[1], ))
        self.drp = DecreaseDropout(dropout, decrease)

        arg = {'kernel_initializer': 'he_uniform'}

        # Build the convolution channel
        model = Reshape((self.inp._keras_shape[1], 1))(self.inp)
        model = Conv1D(64, 16, **arg)(model)
        model = PReLU()(model)
        model = BatchNormalization()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Conv1D(128, 6, **arg)(model)
        model = PReLU()(model)
        model = BatchNormalization()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Conv1D(128, 6, **arg)(model)
        model = PReLU()(model)
        model = BatchNormalization()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = AveragePooling1D(pool_size=2)(model)
        merge = GlobalAveragePooling1D()(model)
        # Defines the merge size
        self.mrg_size = merge._keras_shape[1]

        # Build the encoder
        model = Dense(merge._keras_shape[1], **arg)(merge)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(model._keras_shape[1] // 2, **arg)(model)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        enc_0 = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(model._keras_shape[1] // 2, **arg)(enc_0)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        enc_1 = GaussianNoise(1e-3)(model)
        print('# Latent Space:', enc_1._keras_shape[1])
        
        # Defines the decoder part
        model = Dense(enc_0._keras_shape[1], **arg)(enc_1)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        model = Dense(merge._keras_shape[1], activation='linear', **arg)(model)
        decod = Subtract(name='decode')([merge, model])

        # Defines the output part
        model = Dense(enc_1._keras_shape[1], **arg)(enc_1)
        model = BatchNormalization()(model)
        model = PReLU()(model)
        model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        new = {'activation': 'softmax', 'name': 'output'}
        model = Dense(5, **arg, **new)(model)

        return decod, model
    
    # Model training
    # test_ratio gives 
    # batch_size refers to the used batch_size during learning
    # dropout is the dropout rate into the model
    # patience is the parameter of the EarlyStopping callback
    # verbose indicates whether to use tensorboard or not
    # max_epochs refers to the amount of epochs achievable
    def learn(self, test_ratio=0.3, dropout=0.25, decrease=100, batch_size=64, patience=5, verbose=1, max_epochs=100):

        # Apply data augmentation
        self.bootstrap()

        # Defines the model attributes
        loss = {'output': 'categorical_crossentropy', 'decode': 'mean_squared_error'}
        loss_weights = {'output': 1.0, 'decode': 10.0}
        metrics = {'output': 'accuracy', 'decode': 'mean_absolute_error'}
        monitor = 'val_output_acc'

        # Build the model
        decod, model = self.build(dropout, decrease)
        model = Model(inputs=self.inp, outputs=[model, decod])
        model.compile(loss=loss, metrics=metrics, loss_weights=loss_weights, optimizer='adadelta')

        # Defines the callbacks
        early = EarlyStopping(monitor=monitor, min_delta=1e-5, patience=patience, mode='max')
        check = ModelCheckpoint(self.ate, period=1, monitor=monitor, mode='max', 
                                save_best_only=True, save_weights_only=True)

        # Launch the learning
        his = model.fit(self.raw_t, [np_utils.to_categorical(self.lab_t), np.zeros((len(self.lab_t), self.mrg_size))], 
                        verbose=verbose, epochs=max_epochs, batch_size=batch_size,
                        shuffle=True, callbacks=[self.drp, check, early], validation_split=test_ratio)

        # Save model history
        with open(self.his, 'wb') as raw: pickle.dump(his.history, raw)

        # Memory efficiency
        del model, early, check, his

    # Reconstruct and extract the desired channel
    def get_cv1_channel(self):

        # Recreate the model
        dec, mod = self.build(0.0, 50)
        mod = Model(inputs=self.inp, outputs=[mod, dec])
        mod.load_weights(self.ate)
        # Extract the channel
        mod = Model(inputs=self.inp, outputs=mod.layers[-10].output)

        return mod
