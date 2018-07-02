# Author : Dindin Meryll
# Date : 01/07/2018
# Dreem Headband Sleep Phases Classification Challenge

from package.callback import *

# Defines the data loader
# storage refers to where to pick the data

class Parametrized_Dense:

    # Initialization and load the data
    # params refers to a dictionnary of parameters
    # storage refers to where to pick the data
    # folds refers to the amount of cross-validation rounds
    def __init__(self, params, storage='./dataset/sca_train.h5', folds=5):

        with h5py.File(storage, 'r') as dtb:
            # Load the labels and the corresponding data
            self.lab = dtb['lab'].value.ravel()
            # Define the specific anomaly issue
            self.n_c = len(np.unique(self.l_t))
            # Defines the vectors
            self.fea = dtb['fea'].value

        # Suppress the features with 0 variance
        vtf = VarianceThreshold(threshold=0.0)
        self.fea = vtf.fit_transform(self.fea)

        # Prepares the cross-validation
        self.kfs = KFold(n_splits=folds)
        self.cvs = np.zeros(folds)
        self.prm = params
        self.pth = './models/GA_MOD.weights'

    # Instantiate model and 
    def instantiate_model(self):

        # Defines the callbacks
        arg = {'kernel_initializer': 'he_uniform'}
        ear = EarlyStopping(monitor='val_acc', patience=5, min_delta=1e-5)
        chk = ModelCheckpoint(self.pth, monitor='val_acc', save_best_only=True, save_weights_only=True)
        drp = DecreaseDropout(self.prm['dropout'], 100)
        self.clb = [chk, drp, ear]

        # Defines the model architecture
        self.inp = Input(shape=(X.shape[1], ))
        mod = Dense(self.prm['layer_0'], **arg)(inp)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(drp.prb, drp)(mod)
        mod = Dense(self.prm['layer_1'], **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(drp.prb, drp)(mod)
        mod = Dense(self.prm['layer_2'], **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(drp.prb, drp)(mod)
        mod = Dense(self.prm['layer_3'], **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(drp.prb, drp)(mod)        
        mod = Dense(self.prm['layer_4'], **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(drp.prb, drp)(mod)
        self.mod = Dense(self.n_c, activation='softmax', **arg)(mod)

    # Launch cross-validation
    def scoring(self):

        for idx, (i_t, i_e) in enumerate(self.kfs.split(np.arange(len(self.lab)))):

            # Build the corresponding tuned model
            self.instantiate_model()
            model = Model(inputs=self.inp, outputs=self.mod)
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

            # Launch the training
            model.fit(self.fea[i_t], np_utils.to_categorical(self.lab[i_t]), verbose=0, 
                      epochs=1, shuffle=True, batch_size=self.prm['batchs_size'], 
                      validation_data=(self.fea[i_e], np_utils.to_categorical(self.lab[i_e])))

            # Reload the best model and save the score
            model.load_weights(self.pth)
            prd = [np.argmax(pbs) for pbs in model.predict(self.fea[i_e])]
            self.cvs[idx] = kappa_score(self.lab[i_e], np.asarray(prd).astype('int'))

        return np.mean(self.cvs)

if __name__ == '__main__':

    # Defines the parameters
    params = dict()
    params['dropout'] = float(sys.argv[1])
    params['layer_0'] = int(sys.argv[2])
    params['layer_1'] = int(sys.argv[3])
    params['layer_2'] = int(sys.argv[4])
    params['layer_3'] = int(sys.argv[5])
    params['layer_4'] = int(sys.argv[6])
    params['batch_size'] = int(sys.argv[7])

    warnings.simplefilter('ignore')

    print(-Parametrized_Dense(params).scoring())

