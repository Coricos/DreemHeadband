# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.toolbox import *

# Defines a specific metric for classification

class Metrics(Callback):

    # Initialization
    # autoencoder is a boolean for whether there is an autoencoder or not
    # val_gen refers to the validation generator
    # steps for when the metric must stop the evaluation
    def __init__(self, autoencoder, val_gen, steps):

        super(Callback, self).__init__()

        self.val_gen = val_gen
        self.val_score = []
        self.autoencoder = autoencoder
        self.step = steps

    # Compute the score for each epoch
    # epoch refers to the epoch round
    def on_epoch_end(self, epoch, logs={}):

        # Defines the tools for prediction
        ind, prd, lab = 0, [], []

        for vec in self.val_gen:
            # Iterate according to the right stopping point
            if ind <= self.step :
                lab += [np.argmax(ele) for ele in vec[1]]
                if self.autoencoder: prd += [np.argmax(pbs) for pbs in self.model.predict(vec[0])[0]]
                else: prd += [np.argmax(pbs) for pbs in self.model.predict(vec[0])]
                ind += 1
            else :
                break

        prd, lab = np.asarray(prd), np.asarray(lab)
        kap = kappa_score(lab, prd)
        lin = kappa_score(lab, prd, weights='linear')
        qua = kappa_score(lab, prd, weights='quadratic')
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

# First layer for landscapes ponderation

class SilhouetteLayer(Layer):

    # Initialization
    # output_dim is the desired output dimension
    def __init__(self, output_dim, **kwargs) :
        
        # Defines the output dimension
        self.output_dim = output_dim
        
        super(SilhouetteLayer, self).__init__(**kwargs)

    # Build the layer
    # input_shape refers to the input_shape of the given layer
    def build(self, input_shape) :
        
        # Init all the weights to one
        ini = initializers.Constant(value=1.0)
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(name='kernel', shape=(1, input_shape[-2]), 
                                      initializer=ini, trainable=True)
        
        super(SilhouetteLayer, self).build(input_shape)

    # Fit instance
    # x refers to tf.placeholder
    def call(self, x) :
        
        var = K.reshape(K.sum(K.dot(self.kernel, x), axis=0), (-1, x.get_shape()[-1], 1))
        
        return var

    # Output shape for inference
    # input_shape refers to the input_shape of the given layer
    def compute_output_shape(self, input_shape) :
        
        return (None, self.output_dim, 1)
