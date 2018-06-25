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
            self.raw = dtb[channel].value
        with h5py.File('{}/sca_valid.h5'.format(storage), 'r') as dtb:
            self.raw = np.vstack((self.raw, dtb[channel].value))

    # Build the relative model for drowsiness classification
    # dropout refers to the amount of dropout to set in the model
    # decrease refers to the number of epochs for decreasing ratio
    def build(self, dropout, decrease):

        self.inp = Input(shape=(self.raw.shape[1], ))
        self.drp = DecreaseDropout(dropout, decrease)

        arg = {'kernel_initializer': 'he_uniform'}

        mod = Reshape((self.inp._keras_shape[1], 1))(self.inp)
        mod = Conv1D(64, 50, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = MaxPooling1D(pool_size=5)(mod)
        mod = AdaptiveDropout(self.drp.prb, self.drp)(mod)
        mod = Conv1D(128, 6, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        self.enc = MaxPooling1D(pool_size=5)(mod)
        print('# ENCODER Latent Space', self.enc._keras_shape)

        mod = Conv1D(128, 6, padding='same', **arg)(self.enc)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = UpSampling1D(5)(mod)
        mod = AdaptiveDropout(self.drp.prb, self.drp)(mod)
        mod = Conv1D(64, 50, padding='same', **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = UpSampling1D(5)(mod)
        mod = Conv1D(1, 50, padding='same', **arg)(mod)
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
    def learn(self, test_ratio=0.3, dropout=0.3, decrease=50, batch_size=64, patience=5, verbose=1, max_epochs=100):

        # Build the model
        model = self.build(dropout, decrease)
        model = Model(inputs=self.inp, outputs=model)
        model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adadelta')

        # Defines the callbacks
        early = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=patience, mode='min')
        check = ModelCheckpoint(self.ate, period=1, monitor='val_loss', mode='min', 
                                save_best_only=True, save_weights_only=True)

        # Launch the learning
        his = model.fit(self.raw, self.raw, verbose=verbose, epochs=max_epochs, batch_size=batch_size,
                        shuffle=True, callbacks=[self.drp, check, early], validation_split=test_ratio)

        # Save model history
        with open(self.his, 'wb') as raw: pickle.dump(his.history, raw)

        # Memory efficiency
        del model, early, check, his

    # Reconstruct the model
    def reconstruct(self):

        # Build the model
        self.model = self.build(0.0, 100)
        self.model = Model(inputs=self.inp, outputs=self.ate)
        self.model.load_weights(self.ate)

    # Reconstruct the encoder
    def get_encoder(self):

        if not hasattr(self, 'model'): self.reconstruct()       
        # Serialize the classifier
        model = Model(inputs=model.input, outputs=model.layers[-10].output)

        return model
