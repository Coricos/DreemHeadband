# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.database import *
from package.callback import *

# Defines the specific network strategy for EEG signals

class DS_Model:

    # Initialization
    # input_db refers to where to gather the data
    # channels refers to which EEG channel to look for
    # marker refers to the model ID
    def __init__(self, input_db, channel, marker=None):

        self.pth = input_db
        self.cls = channel
        # Output definition
        if marker: self.spc = './models/SPC_{}_{}.weights'.format(self.cls, marker)
        else: self.spc = './models/SPC_{}.weights'.format(self.cls)
        if marker: self.out = './models/DSN_{}_{}.weights'.format(self.cls, marker)
        else: self.out = './models/DSN_{}.weights'.format(self.cls)
        if marker: self.his = './models/HIS_DSN_{}_{}.history'.format(self.cls, marker)
        else: self.his = './models/HIS_DSN_{}.history'.format(self.cls)

        # Handling labels
        with h5py.File(self.pth, 'r') as dtb:
            self.l_t = dtb['lab_t'].value.ravel()
            self.l_e = dtb['lab_e'].value.ravel()
            self.n_c = len(np.unique(self.l_t))
        # Load the needed data
        with h5py.File(self.pth, 'r') as dtb:
            self.train = [dtb['eeg_{}_t'.format(self.cls)].value, dtb['eeg_{}_t'.format(self.cls)].value]
            self.valid = dtb['eeg_{}_e'.format(self.cls)].value
            self.evals = dtb['eeg_{}_v'.format(self.cls)].value
            self.sfreq = self.train.shape[1] // 30

    # Build the tensorflow graph for spatial learning
    # inp refers to the tensorflow input
    # drp refers to the annealing dropout callback
    def build_spatial(self, inp, drp):

        # Layers arguments
        arg = {'kernel_initializer': 'he_normal'}

        # Build the channel aimed at small patterns
        s_mod = Reshape((inp._keras_shape[1], 1))(inp)
        s_mod = Conv1D(64, self.sfreq//2, strides=self.sfreq//16, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = MaxPooling1D(pool_size=8, strides=8)(s_mod)
        s_mod = AdaptiveDropout(drp.prb, drp)(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = Conv1D(128, 8, **arg)(s_mod)
        s_mod = PReLU()(s_mod)
        s_mod = BatchNormalization()(s_mod)
        s_mod = MaxPooling1D(pool_size=4, strides=4)(s_mod)

        # Build the channel aimed at longer patterns
        l_mod = Reshape((inp._keras_shape[1], 1))(inp)
        l_mod = Conv1D(64, self.sfreq*4, strides=13, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = MaxPooling1D(pool_size=4, strides=4)(l_mod)
        l_mod = AdaptiveDropout(drp.prb, drp)(l_mod)
        l_mod = Conv1D(128, 6, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = Conv1D(128, 6, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = Conv1D(128, 6, **arg)(l_mod)
        l_mod = PReLU()(l_mod)
        l_mod = BatchNormalization()(l_mod)
        l_mod = MaxPooling1D(pool_size=2, strides=2)(l_mod)

        # Concatenate both channels
        space = concatenate([s_mod, l_mod])
        space = AdaptiveDropout(drp.prb, drp)(space)

        # Defines last layer for training purposes
        model = GlobalMaxPooling1D()(space)
        model = Dense(self.n_c, activation='softmax', **arg)(model)

        return model

    # Trains a first spatial representation of the EEG signals
    # epochs refers to the maximum amount of epochs
    # batch refers to the batch_size
    # ini_dropout refers to the initialization of the annealing dropout
    # decrease refers to the amount of epochs before annealation of the dropout
    def train_spatial(self, epochs=200, batch=128, ini_dropout=0.5, decrease=100):

        # Prepares the data, which will be balanced through oversampling
        ros = RandomOverSampler()
        vec, lab = shuffle(*ros.fit_sample(self.train, self.l_t))

        # Layers arguments
        drp = DecreaseDropout(ini_dropout, decrease)
        ear = EarlyStopping(monitor='loss', min_delta=1e-5, patience=10, verbose=0)
        chk = ModelCheckpoint(self.spc, monitor='loss', save_best_only=True, save_weights_only=True)

        # Build the model
        inp = Input(shape=(vec.shape[1], ))
        model = self.build_spatial(inp, drp)

        # Launch the learning process
        model = Model(inputs=inp, outputs=model)
        optim = Adadelta(clipnorm=1.0)
        model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=optim)
        model.fit(vec, np_utils.to_categorical(lab), verbose=1, epochs=epochs,
                  callbacks=[drp, ear, chk], shuffle=True, validation_split=0.0,
                  class_weight=class_weight(lab.ravel()), batch_size=batch)

        # Memory efficiency
        del ros, vec, lab, inp, drp, ear, chk, model, optim
    
    # Build the tensorflow graph for temporal learning
    # inp refers to the tensorflow input
    # drp refers to the annealing dropout callback
    def build_temporal(self, inp, drp):

        # Layers argument
        arg = {'kernel_initializer': 'he_normal',
               'kernel_regularizer': regularizers.l2(1e-3)}

        # Load the spatial model
        space = self.build_spatial(inp, drp)
        space = Model(inputs=inp, outputs=space)
        space.load_weights(self.spc)
        space = Model(inputs=space.input, outputs=space.layers[-3].output)
        
        # Intialize the new model
        model = Reshape((inp._keras_shape[1], 1))(inp)
        model = space(model)

        tempo = Bidirectional(LSTM(512, return_sequences=True, **arg))(model)
        tempo = BatchNormalization()(tempo)
        tempo = PReLU()(tempo)
        tempo = AdaptiveDropout(drp.prb, drp)(tempo)
        tempo = Bidirectional(LSTM(512, return_sequences=False, **arg))(tempo)
        tempo = BatchNormalization()(tempo)
        tempo = PReLU()(tempo)
        tempo = AdaptiveDropout(drp.prb, drp)(tempo)

        # Residual fully connected layer
        resid = GlobalMaxPooling1D()(model)
        resid = Dense(1024, **arg)(resid)
        resid = BatchNormalization()(resid)
        resid = PReLU()(resid)

        model = Add()([tempo, resid])
        model = AdaptiveDropout(drp.prb, drp)(model)
        model = Dense(self.n_c, activation='softmax', **arg)(model)

        return model        

    # Ends the training by merging the spatial features into a temporal dimension
    # epochs refers to the maximum amount of epochs
    # batch refers to the batch_size
    # ini_dropout refers to the initialization of the annealing dropout
    # decrease refers to the amount of epochs before annealation of the dropout
    def train_temporal(self, epochs=200, batch=32, ini_dropout=0.5, decrease=50):

        # Prepares the data, which will be balanced through oversampling
        ros = RandomOverSampler()
        vec, lab = shuffle(*ros.fit_sample(self.train, self.l_t))

        # Layers arguments
        drp = DecreaseDropout(ini_dropout, decrease)
        ear = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=0)
        chk = ModelCheckpoint(self.out, monitor='val_loss', save_best_only=True, save_weights_only=True)
        
        # Build the input
        inp = Input(shape=(self.train.shape[1], ))
        model = self.build_temporal(inp, drp)

        # Launch the learning process
        model = Model(inputs=inp, outputs=model)
        optim = Adam(lr=1e-4, clipnorm=1.0)
        model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=optim)
        his = model.fit(self.train, np_utils.to_categorical(self.l_t), verbose=1, epochs=epochs,
                        callbacks=[drp, ear, chk], validation_data=(self.valid, np_utils.to_categorical(self.l_e)),
                        class_weight=class_weight(self.l_t.ravel()), batch_size=batch, shuffle=True)

        # Serialize the learning history
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

    # Rebuild the tensorflow graph for temporal learning
    # inp refers to the tensorflow input
    # drp refers to the annealing dropout callback
    def load_model(self, inp, drp):

        # Build, select and truncate the temporal model
        model = self.build_temporal(inp, drp)
        model = Model(inputs=inp, outputs=model)
        model.load_weights(self.out)
        model = Model(inputs=model.input, outputs=model.layers[-5].output)

        return model
