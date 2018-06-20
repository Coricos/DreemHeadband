# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.toolbox import *

# Defines a specific metric for classification

class Metrics(Callback):

    # Initialization
    # autoencoder is a boolean for whether there is an autoencoder or not
    # l_e refers to the correct labels of testing set
    # val_gen refers to the validation generator
    def __init__(self, autoencoder, l_e, val_gen):

        super(Callback, self).__init__()

        self.l_e = l_e
        self.val_gen = val_gen
        self.val_score = []
        self.autoencoder = autoencoder

    # Compute the score for each epoch
    # epoch refers to the epoch round
    def on_epoch_end(self, epoch, logs={}):

        # Defines the size of the validation set
        sze = len(self.l_e)
        # Defines the tools for prediction
        ind, prd = 0, []

        for vec in self.val_gen:
            # Defines the right stop according to the batch_size
            if (sze / 512) - int(sze / 512) == 0 : end = int(sze / 512) - 1
            else : end = int(sze / 512)
            # Iterate according to the right stopping point
            if ind <= end :
                if self.autoencoder: prd += [np.argmax(pbs) for pbs in self.model.predict(vec)[0]]
                else: prd += [np.argmax(pbs) for pbs in self.model.predict(vec)]
                ind += 1
            else : 
                break

        prd = np.asarray(prd)
        kap = kappa_score(self.l_e, prd)
        lin = kappa_score(self.l_e, prd, weights='linear')
        qua = kappa_score(self.l_e, prd, weights='quadratic')
        self.val_score.append(kap)
    
        print(' - kappa_score: %f - kappa_linear: %f - kappa_quadra: %f' % (kap, lin, qua))

        return

# Defines the callback for merge on the adaptive dropout

class DecreaseDropout(Callback):

    # Initialization
    def __init__(self, prb, steps):

        super(Callback, self).__init__()

        self.ini = prb
        self.prb = prb
        self.steps = steps

    # Changes the dropout rate depending on the epoch
    def on_epoch_end(self, epoch, logs=None):

        self.prb = max(0, 1 - epoch/self.steps) * self.ini

        return

# Layer corresponding to the adaptive dropout

class AdaptiveDropout(Layer):

    # Initialization
    def __init__(self, p, callback, **kwargs):

        self.p = p
        self.callback = callback
        if 0. < self.p < 1.: self.uses_learning_phase = True
        self.supports_masking = True

        super(AdaptiveDropout, self).__init__(**kwargs)

    # Defines the core operation of the layer
    def call(self, x, mask=None):

        self.p = self.callback.prb

        if 0. < self.p < 1.:
            x = K.in_train_phase(K.dropout(x, level=self.p), x)

        return x

    # Returns config for serialization
    def get_config(self):

        config = {'p': self.p}
        base_config = super(AdaptiveDropout, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# Aims at shuffling the data for better generalization

class DataShuffler(Callback):

    # Initialization
    # dtb_path refers to the database path
    # rounds refers to the amount of rounds for which the data is shuffled
    def __init__(self, dtb_path, rounds):

        super(Callback, self).__init__()

        self.pth = dtb_path
        self.rnd = rounds

    # Shuffles the data at each end of epoch
    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.rnd == 0:

            with h5py.File(self.pth, 'a') as dtb:

                i_t = shuffle(np.arange(dtb['lab_t'].shape[0]))
                for key in [ele for ele in dtb.keys() if ele[-1] == 't']:
                    dtb[key][...] = dtb[key].value[i_t]

        else: pass

        return
