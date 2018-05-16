# DINDIN Meryll
# April 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.toolbox import *

class Database:

    def __init__(self, storage='./dataset'):

        self.train_pth = '{}/train.h5'.format(storage)
        self.valid_pth = '{}/valid.h5'.format(storage)

        self.storage = storage
        self.sets_size = []
        
        with h5py.File(self.train_pth, 'r') as dtb:
            self.sets_size.append(dtb['po_r'].shape[0])
            self.keys = list(dtb.keys())
        with h5py.File(self.valid_pth, 'r') as dtb:
            self.sets_size.append(dtb['po_r'].shape[0])

    def build(self, vec_size=2000, out_storage='/mnt/Storage'):

        # Filtering through Kalman filter
        train_out = '{}/dts_train.h5'.format(out_storage)
        valid_out = '{}/dts_valid.h5'.format(out_storage)

        # Defines the parameters for each key
        fil = {'po_r': True, 'po_ir': True,
               'acc_x': False, 'acc_y': False, 'acc_z': False,
               'eeg_1': True, 'eeg_2': True, 'eeg_3': True, 'eeg_4': True}
        dic = {'eeg_1': (4, 20), 'eeg_2': (4, 20), 'eeg_3': (4, 20),
               'eeg_4': (4, 20), 'po_ir': (3, 5), 'po_r': (3, 5)}

        # Iterates over the keys
        for key in tqdm.tqdm(fil.keys()):
            # Link inputs to outputs
            for out, pth in zip([train_out, valid_out], [self.train_pth, self.valid_pth]):
                # Load the values
                with h5py.File(pth, 'r') as dtb: val = dtb[key].value

                # Apply the kalman filter if needed
                if fil[key]:
                    pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    arg = {'std_factor': dic[key][0], 'smooth_window': dic[key][1]}
                    fun = partial(kalman_filter, **arg)
                    val = np.asarray(pol.map(fun, val))
                    pol.close()
                    pol.join()

                # Adapt the size of the vectors
                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                fun = partial(interpolate, size=vec_size)
                val = np.asarray(pol.map(fun, val))
                pol.close()
                pol.join()

                # Serialize the outputs
                with h5py.File(out, 'a') as dtb:
                    if dtb.get(key): del dtb[key]
                    dtb.create_dataset(key, data=val)

                # Memory efficiency
                del pol, fun, val, arg

        self.train_pth = train_out
        self.valid_pth = valid_out

    def redirect(self, out_storage='/mnt/Storage'):

        self.train_pth = '{}/dts_train.h5'.format(out_storage)
        self.valid_pth = '{}/dts_valid.h5'.format(out_storage)

        with h5py.File(self.train_pth, 'r') as dtb:
            self.keys = list(dtb.keys())

    def add_norm(self):

        for pth in [self.train_pth, self.valid_pth]:

            with h5py.File(pth, 'r') as dtb:

                tmp = np.square(dtb['acc_x'].value)
                tmp += np.square(dtb['acc_y'].value)
                tmp += np.square(dtb['acc_z'].value)

            with h5py.File(pth, 'a') as dtb:

                if dtb.get('norm'): del dtb['norm']
                dtb.create_dataset('norm', data=tmp)

            # Memory efficiency
            del tmp

    def add_FFT(self, n_components=50):

        for pth in [self.train_pth, self.valid_pth]:
        
            for key in ['norm', 'po_ir', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:

                with h5py.File(pth, 'r') as dtb: val = dtb[key].value

                pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                fun = partial(compute_fft, n_components=n_components)
                fft = pol.map(fun, val)
                pol.close()
                pol.join()

                with h5py.File(pth, 'a') as dtb: 
                    dtb.create_dataset('fft_{}'.format(key), data=val)

                del pol, fun, fft

    def add_PCA(self, n_components=5):

        res = np.empty((np.sum(self.sets_size), len(self.KEYS)*n_components))

        for ind, key in tqdm.tqdm(enumerate(self.KEYS)):

            pca = PCA(n_components=n_components)

            for pth in [self.train_pth, self.valid_pth]:
                
                with h5py.File(pth, 'r') as dtb:
                    pca.partial_fit(dtb[key].value)

            for pth in [self.train_pth, self.valid_pth]:

                tmp = []
                with h5py.File(pth, 'r') as dtb:
                    tmp.append(pca.transform(dtb[key].value))

            res[:,5*ind:5*(ind+1)] = np.vstack(tuple(tmp))

            del pca, tmp

        with h5py.File(self.train_pth, 'a') as dtb:
            dtb.create_dataset('pca', data=res[:self.sets_size[0]])
        with h5py.File(self.valid_pth, 'a') as dtb:
            dtb.create_dataset('pca', data=res[-self.sets_size[1]:])

        self.keys.append('pca')

    def add_CHAOS(self):

        for idx, pth in enumerate([self.train_pth, self.valid_pth]):

            res = np.empty(self.sets_size[idx], len(self.KEYS)*6)

            for ind, key in tqdm.tqdm(enumerate(self.KEYS)):

                apl = [nolds.sampen, nolds.dfa, nolds.hurst, nolds.lyap_r]

                for inc, fun in enumerate(apl):
                    
                    with h5py.File(pth, 'r') as dtb:
                        tmp = np.apply_along_axis(fun, 1, dtb[key].value)
                        res[:,ind+apl] = tmp
                    
            with h5py.File(pth, 'a') as dtb:
                dtb.create_dataset('chaos', data=res)

            self.keys.append('chaos')

    def rescale(self):

        for key in tqdm.tqdm(['acc_x', 'acc_y', 'acc_z', 'norm', 'eeg_1', 
                              'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):

            mms = MinMaxScaler(feature_range=(0, 1))
            sts = StandardScaler(with_std=False)

            for pth in [self.train_pth, self.valid_pth]:

                with h5py.File(pth, 'r') as dtb:
                    mms.partial_fit(np.hstack(dtb[key].value).reshape(-1,1))

            for pth in [self.train_pth, self.valid_pth]:

                with h5py.File(pth, 'r') as dtb:
                    shp = dtb[key].shape
                    tmp = mms.transform(np.hstack(dtb[key].value).reshape(-1,1))
                    sts.partial_fit(tmp)
                    del shp, tmp

            pip = Pipeline([('mms', mms), ('sts', sts)])

            for pth in [self.train_pth, self.valid_pth]:

                with h5py.File(pth, 'a') as dtb:
                    shp = dtb[key].shape
                    val = pip.transform(np.hstack(dtb[key].value).reshape(-1,1)).reshape(shp)
                    dtb[key][...] = val

        for key in tqdm.tqdm(['pca', 'chaos', 'fft_norm', 'fft_po_ir',
                              'fft_eeg_1', 'fft_eeg_2', 'fft_eeg_3', 'fft_eeg_4']):

            try: 
                mms = MinMaxScaler(feature_range=(0, 1))

                for pth in [self.train_pth, self.valid_pth]:

                    with h5py.File(pth, 'r') as dtb:
                        mms.partial_fit(dtb[key].value)

                for pth in [self.train_pth, self.valid_pth]:

                    with h5py.File(pth, 'a') as dtb:
                        dtb[key][...] = mms.transform(dtb[key].value)

            except:
                pass
