# DINDIN Meryll
# Nov 16th, 2018
# Dreem Headband Sleep Phases Classification Challenge

try: from package.toolbox import *
except: from toolbox import *

# Develops a class relative to profile extraction

class Profiles:

    def __init__(self, labels):

        try: self.lab = labels.values.ravel()
        except: self.lab = labels.ravel()

    def ratios(self):

        prp = [len(np.where(self.lab == idx)[0]) / len(self.lab) for idx in np.unique(self.lab)]

        plt.figure(figsize=(18,4))
        plt.bar(np.arange(len(prp)), prp, color='salmon')
        plt.xticks(np.arange(len(prp)), ['Awake', 'Stage 1', 'Stage 2', 'Stage 3 & 4', 'REM Sleep'])
        plt.ylabel('Relative Ratio')
        plt.show()

    def build_profiles(self):

        lst = [list(grp) for k, grp in groupby(self.lab)]
        s_p = np.asarray([ele[0] for ele in lst])
        sms = np.asarray([len(ele) for ele in lst])
        ind = np.where(s_p == 0)[0]

        idx = np.asarray(sorted(list(set(np.where(sms > 10)[0]) & set(ind))))
        mea = np.mean(np.diff(np.cumsum(sms)[idx]) * 30) / 60 / 60

        prf, mkr = [], []
        for pat in range(len(idx)-1):
            dec = np.asarray([0] + list(np.cumsum(sms)))
            beg, end = max(0, dec[idx[pat]] + sms[idx[pat]] // 2), min(len(self.lab), dec[idx[pat+1]] + sms[idx[pat+1]] // 2)
            mkr.append((beg, end))
            prf.append(self.lab[beg:end])
            
        return prf, mkr

    def display_profile(self, profile):

        plt.figure(figsize=(18,4))
        ind = np.arange(len(profile))
        for idx in np.unique(profile):
            plt.hlines(idx, 0, len(profile), linestyles='--', lw=1.0)
            ext = np.where(profile == idx)[0]
            plt.bar(ind[ext], 0.5, bottom=profile[ext]-0.25, width=1.2)
        plt.ylim([-0.5, 4.5])
        plt.xlabel('Time')
        plt.ylabel('Sleep Phase')
        plt.show()

    def learn(self, profiles, save='./models/lstm_labels.ks', timesteps=30, test_size=0.25, epochs=100):

        vec = []

        # Decompose the profiles into learning timesteps
        for profile in profiles:
            m_i = np.arange(timesteps+1)*(0+1)
            m_j = np.arange(np.max(profile.shape[0]-(timesteps)*(0+1), 0))
            vec.append(profile[m_i + m_j.reshape(-1,1)])
        # Categorical transformation
        vec = np_utils.to_categorical(np.vstack(tuple(vec)))
        x,y = vec[:,:timesteps,:], vec[:,-1,:]
        # Train test separation
        x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=test_size, shuffle=True)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(30, input_shape=(x_t.shape[1], x_t.shape[2]), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(30, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # Defines the callbacks
        arg = {'patience': 5, 'verbose': 0}
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, **arg)
        arg = {'save_best_only': True, 'save_weights_only': True}
        check = ModelCheckpoint(save, monitor='val_acc', **arg)

        # Launch the training
        model.fit(x_t, y_t, epochs=epochs, batch_size=32, shuffle=True, 
                  callbacks=[early, check], validation_data=(x_v, y_v))

# Detect anomalies

class Anomaly:

    def __init__(self, h5_train='./dataset/train.h5', h5_valid='./dataset/valid.h5'):

        e_key = ['eeg_{}'.format(dim) for dim in range(1, 5)]

        with h5py.File(h5_train, 'r') as dtb:
            eeg_t = np.asarray([dtb[key].value for key in e_key])
            nrm_t = np.sqrt(np.sum(np.square(eeg_t), axis=0))
            del eeg_t

        with h5py.File(h5_valid, 'r') as dtb:
            eeg_v = np.asarray([dtb[key].value for key in e_key])
            nrm_v = np.sqrt(np.sum(np.square(eeg_v), axis=0))
            del eeg_v

        self.nrm_s = np.vstack((nrm_t, nrm_v))
        del nrm_t, nrm_v

    def spot_outliers(self, update=True, save='./models/row_mask.npy', threshold=4):

        msk = np.ones(len(self.nrm_s), dtype=bool)
        # Build standard features for outliers spotting
        m_x = np.max(self.nrm_s, axis=1)
        grd = np.max(np.gradient(self.nrm_s, axis=1), axis=1)
        auc = np.trapz(self.nrm_s, axis=1)
        # Fill the mask
        msk[outlier_from_median(m_x, threshold)] = False
        msk[outlier_from_median(grd, threshold)] = False
        msk[outlier_from_median(auc, threshold)] = False

        np.save(save, msk)

        return msk
