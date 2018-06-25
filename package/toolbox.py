# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.topology import *

# Defines a function to rename the datasets for clearer management
# storage refers to where to pick the dataset
def rename(storage='./dataset'):

    train_pth = '{}/train.h5'.format(storage)
    valid_pth = '{}/valid.h5'.format(storage)

    for pth in [train_pth, valid_pth]:

        with h5py.File(pth, 'a') as dtb:

            dtb['acc_x'] = dtb['accelerometer_x']
            del dtb['accelerometer_x']
            dtb['acc_y'] = dtb['accelerometer_y']
            del dtb['accelerometer_y']
            dtb['acc_z'] = dtb['accelerometer_z']
            del dtb['accelerometer_z']

# Display a specific 30s sample
# idx refers to the index of the sample
def display(idx, storage='./dataset/valid.h5'):

    with h5py.File(storage, 'r') as dtb:
        # Load the signals
        a_x = dtb['acc_x'][idx,:]
        a_y = dtb['acc_y'][idx,:]
        a_z = dtb['acc_z'][idx,:]
        n_a = np.sqrt(np.square(a_x) + np.square(a_y) + np.square(a_z))
        e_1 = dtb['eeg_1'][idx,:]
        e_2 = dtb['eeg_2'][idx,:]
        e_3 = dtb['eeg_3'][idx,:]
        e_4 = dtb['eeg_4'][idx,:]
        o_i = dtb['po_ir'][idx,:]
        o_r = dtb['po_r'][idx,:]

    # Defines the figure
    plt.figure(figsize=(18,10))
    fig = gd.GridSpec(4, 12)
    plt.subplot(fig[0, 0:4])
    plt.plot(a_x, label='Acc_X')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[0, 4:8])
    plt.plot(a_y, label='Acc_Y')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[0, 8:12])
    plt.plot(a_z, label='Acc_X')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[1, :])
    plt.plot(n_a, label='Normed_Acc')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[2, 0:3])
    plt.plot(e_1, label='EEG_1')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[2, 3:6])
    plt.plot(e_2, label='EEG_2')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[2, 6:9])
    plt.plot(e_3, label='EEG_3')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[2, 9:12])
    plt.plot(e_4, label='EEG_4')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[3, 0:6])
    plt.plot(o_i, label='Oxygen_IR')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(fig[3, 6:12])
    plt.plot(o_r, label='Oxygen_R')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Defines the kalman filter for noise reduction
# std_factor refers to the sought reduction of deviation of the signal
# smooth_window refers to the convolution window for smoothing
def kalman_filter(val, std_factor=3, smooth_window=5):

    if np.std(val) < 1e-5: 
        return np.zeros(len(val))

    else:
        # Initialize the arrays
        x_t = np.zeros(val.shape[0])
        P_t = np.zeros(val.shape[0])
        x_m = np.zeros(val.shape[0])
        P_m = np.zeros(val.shape[0])
        fac = np.zeros(val.shape[0])
        # Defines the variables
        R = (np.std(val))**2
        Q = (np.std(val) / std_factor)**2
        tmp = np.nanmean(val[:5])
        if np.isnan(tmp): x_t[0] = np.nanmean(val)
        else: x_t[0] = tmp    
        P_t[0] = np.std(val)

        # Iterative construction
        for k in range(1, val.shape[0]):
            x_m[k] = x_t[k-1]
            P_m[k] = P_t[k-1] + Q
            fac[k] = P_m[k] / (P_m[k] + R)
            x_t[k] = x_m[k] + fac[k] * (val[k] - x_m[k])
            P_t[k] = (1 - fac[k]) * P_m[k]

        # Apply smoothing
        b = np.full(smooth_window, 1.0 / smooth_window)
        x_t = sg.lfilter(b, 1, x_t)

        # Memory efficiency
        del P_t, x_m, P_m, fac, R, Q
        
        return x_t

# Resize the 30s epochs for better understanding through convolution
# size refers to the ending lenght
def resize_time_serie(val, size=400, threshold=1.0):

    if np.max(np.abs(val)) > threshold:
        mms = MinMaxScaler(feature_range=(0,2*threshold))
        sts = StandardScaler(with_std=False)
        sca = Pipeline([('mms', mms), ('sts', sts)])
    else:
        sca = StandardScaler(with_std=False)
    
    return sca.fit_transform(interpolate(val, size=400).reshape(-1,1)).ravel()

# Defines a vector reduction through interpolation
# val refers to a 1D array
# size refers to the desired size
def interpolate(val, size=2000):

    # Defines whether the size is determined or not
    x = np.linspace(0, size, num=len(val), endpoint=True)
    o = np.linspace(0, size, num=size, endpoint=True)

    if len(val) < size:
        f = interp1d(x, val, kind='linear', fill_value='extrapolate')
        return f(o)

    if len(val) == size:
        return val

    if len(val) > size:
        f = interp1d(x, val, kind='cubic')
        return f(o)

# Compute features related to the chaos theory
# val refers to a 1D array
def compute_tda_features(val):
    
    res, fil = [], Levels(val)

    try: 
        u,d = fil.get_persistence()
        for ele in u, d:
            dig = ele[:,1] - np.sum(ele, axis=1)/2
            res += [np.max(dig), np.mean(dig), np.std(dig)]
            for per in [25, 50, 75, 90]: res.append(np.percentile(dig, per))
    except: 
        res += list(np.zeros(14))

    try: 
        p,q = fil.betti_curves()
        for ele in p, q:
            res.append(np.trapz(ele))
            res.append(np.max(ele))
    except: 
        res += list(np.zeros(4)) 

    try: 
        r,s = fil.landscapes()
        for ele in r, s:
            for ldc in ele: 
                res.append(np.trapz(ldc))
                res.append(np.max(ldc))
    except:
        res += list(np.zeros(40))
            
    del fil
    
    return np.asarray(res)

# Apply a logarithmic envelope to a whole signal
# val refers to a 1D array
# coeff refers to the upper dimension reduction
def envelope(val, m_x=1.0, coeff=2.0):
    
    tmp = np.zeros(len(val))
    idx = np.where(val > 0)[0]
    tmp[idx] = np.log(1 + coeff*val[idx] / m_x) / coeff
    idx = np.where(val < 0)[0]
    tmp[idx] = - np.log(1 + coeff*np.abs(val[idx]) / m_x) / coeff
    
    return tmp

# Defines a dictionnary composed of class weights
# lab refers to a 1D array of labels
def class_weight(lab) :
    
    res = dict()
    wei = compute_class_weight('balanced', np.unique(lab), lab)
    for idx, ele in enumerate(wei) : res[idx] = ele
        
    return res

# Defines a vectorization process for sliding window
# vec_size refers to the size of the output vector
# overlap refers to the amount of needed overlap
def vectorization(val, vec_size, overlap):

    stp = int((1-overlap) * vec_size)
    
    new, ind = [], 0
    while ind + vec_size < len(val):
        new.append((ind, ind + vec_size))
        ind += stp
    
    return np.asarray([val[i:j] for i,j in new])

# Needed for imbalance counter-balancing
# lab refers to a 1D array of labels
def sample_weight(lab) :

    # Defines the sample_weight
    res = np.zeros(len(lab))
    wei = compute_class_weight('balanced', np.unique(lab), lab)
    wei = wei / sum(wei)
    
    for ele in np.unique(lab) :
        for idx in np.where(lab == ele)[0] :
            res[idx] = wei[int(ele)]

    del wei

    return res

# Defines the scoring function
# true refers to the true labels
# pred refers to the predicted labels
# weights refers to the boolean activation
def kappa_score(true, pred, weights=None):

    cfm = confusion_matrix(true, pred)
    n_c = len(np.unique(true))
    s_0 = np.sum(cfm, axis=0)
    s_1 = np.sum(cfm, axis=1)
    exp = np.outer(s_0, s_1).astype(np.double) / np.sum(s_0) 
    mat = np.ones([n_c, n_c], dtype=np.int)
    
    if weights == 'linear':

        mat += np.arange(n_c)
        mat = np.abs(mat - mat.T)

    elif weights == 'quadratic':

        mat += np.arange(n_c)
        mat = (mat - mat.T) ** 2

    else: 

        mat.flat[::n_c + 1] = 0

    sco = np.sum(mat * cfm) / np.sum(mat * exp)
    
    return 1 - sco

# Reset the mean of independent vectors
# vec refers to a 1D array
def reset_mean(vec):

    return StandardScaler(with_std=False).fit_transform(vec.reshape(-1,1)).ravel()

# Compute the Betti curves
# vec refers to a 1D array
def compute_betti_curves(vec):

    fil = Levels(vec)
    try: v,w = fil.betti_curves(num_points=100)
    except: v,w = np.zeros(100), np.zeros(100)
    del fil
    
    return np.vstack((v,w))

# Compute the landscapes
# vec refers to a 1D array
def compute_landscapes(vec):

    fil = Levels(vec)
    try: p,q = fil.landscapes(num_points=100)
    except: p,q = np.zeros((10,100)), np.zeros((10,100))
    del fil
    
    return np.vstack((p,q))

# Defines the feature construction pipeline
# val refers to a 1D array
def compute_features(val):

    # Defines the amount of crossing-overs
    def crossing_over(val):
        
        sgn = np.sign(val)
        sgn[sgn == 0] == -1

        return len(np.where(np.diff(sgn))[0])
    
    # Defines the entropy of the signal
    def entropy(val):
        
        dta = np.round(val, 4)
        cnt = Counter()

        for ele in dta: cnt[ele] += 1

        pbs = [val / len(dta) for val in cnt.values()]
        pbs = [prd for prd in pbs if prd > 0.0]

        ent = 0.0
        for prd in pbs:
            ent -= prd * log(prd, 2.0)

        return ent

    # Defines the whole fourier features
    def fourier(val, n_features=25):

        res = []

        fft = np.abs(np.fft.rfft(val))
        for per in [25, 50, 75]: res.append(np.percentile(fft, per))
        f,s = sg.periodogram(val, fs=len(val) / 30)
        res.append(f[s.argmax()])
        res.append(np.max(s))
        res.append(np.sum(s))
        res.append(entropy(s))

        return res

    # Defines the wavelet features
    def wavelet(val):

        res = []

        c,_ = pywt.cwt(val, np.arange(1, 32), 'cmor', 30)
        coe = (np.square(np.abs(c))).mean(axis=0)
        res.append(np.trapz(coe))
        res.append(np.mean(coe))
        res.append(np.std(coe))
        res.append(entropy(coe))
        for per in [25, 50, 75]: 
            res.append(np.percentile(coe, per))

        cA_5, cD_5, cD_4, cD_3, _, _ = pywt.wavedec(val, 'db4', level=5)
    
        for sig in [cA_5, cD_5, cD_4, cD_3]:
            res += [np.min(sig), np.max(sig), np.sum(np.square(sig)), np.mean(sig), np.std(sig)]
            
        sgn = np.sign(val)
        sgn = np.split(sgn, np.where(np.diff(sgn) != 0)[0]+1)
        sgn = np.asarray([len(ele) for ele in sgn])
        res += [np.nanmean(sgn), np.std(sgn)]
        
        sgn, ine = np.asarray([0] + list(sgn)), 0.0
        for idx in range(len(sgn)-1):
            ine += (sgn[idx+1] - sgn[idx])*np.trapz(np.abs(val[np.sum(sgn[:idx]):np.sum(sgn[:idx+1])]))
        res.append(ine)

        return res

    # Compute the spectrogram features
    def spectrogram(val):

        res = []
        f_s = len(val)/30

        f,_,S = sg.spectrogram(val, fs=f_s, return_onesided=True)
        res += list(f[S.argmax(axis=0)])
        res += list(np.max(S, axis=0))
        psd = np.sum(S, axis=0)
        res += list(psd)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

        f,_,Z = sg.stft(val, fs=f_s, window='hamming', nperseg=int(5*f_s), noverlap=int(0.7*5*f_s))
        Z = np.abs(Z.T)
        res += list(f[Z.argmax(axis=1)])
        res += list(np.max(Z, axis=1))
        psd = np.sum(Z, axis=1)
        res += list(psd)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

        return res

    # Compute hjorth features
    def neural_features(val):

        res = []
        
        try:
            tmp = neurokit.complexity(val, sampling_rate=len(val)/30, lyap_r=True, lyap_e=True)
            for key in sorted(list(tmp.keys())): res.append(tmp[key])
        except:
            res += list(np.zeros(16))

        arg = {'shannon': False, 'sampen': False, 'multiscale': False, 'svd': False, 'correlation': False, 
               'higushi': False, 'petrosian': False, 'fisher': False, 'hurst': False, 'dfa': False}
        for bands in [[0.5, 4, 8, 13, 30], [0.5, 1.5, 12, 14]]:
            try: res.append(list(neurokit.complexity(val, sampling_rate=len(val)/30, bands=bands, **arg).values())[0])
            except: res.append(0.0)

        dif = np.diff(val)
        dif = np.asarray([val[0]] + list(dif))
        num = len(val)

        m_2 = float(np.sum(dif ** 2)) / num
        t_p = np.sum(np.square(val))
        res.append(np.sqrt(m_2 / t_p))
        
        m_4 = 0.0
        for idx in range(1, len(dif)):
            m_4 += np.square(dif[idx] - dif[idx-1])
        m_4 = m_4 / num
        res.append(np.sqrt(m_4 * t_p / m_2 / m_2))

        return np.asarray(res)

    res = []

    # Build the feature vector
    res.append(np.mean(val))
    res.append(np.std(val))
    res.append(min(val))
    res.append(max(val))
    for per in [25, 50, 75]: res.append(np.percentile(val, per))
    # Frequential features
    res += list(fourier(val))
    # Wavelet features
    res += list(wavelet(val))
    # Spectrogram features
    res += list(spectrogram(val))
    # Statistical features
    res.append(kurtosis(val))
    res.append(skew(val))
    res.append(crossing_over(val))
    res.append(entropy(val))
    res += list(neural_features(val))
    res += list(compute_tda_features(val))
    # Features over gradient
    grd = np.gradient(val)
    res.append(np.mean(grd))
    res.append(np.std(grd))
    res.append(min(grd))
    res.append(max(grd))
    res.append(entropy(grd))

    return np.asarray(res)

# Easier to call and recreate the channel array
# turn_on refers to the list of channels to turn-on
def generate_channels(turn_on):

    dic = {
           'with_acc_cv2': False,
           'with_acc_cv1': False,
           'with_acc_cvl': False,
           'with_n_a_cv1': False,
           'with_n_a_cvl': False,
           'with_eeg_cv2': False,
           'with_eeg_cv1': False,
           'with_eeg_cvl': False,
           'with_eeg_tda': False,
           'with_eeg_enc': False,
           'with_eeg_ate': False,
           'with_eeg_l_0': False,
           'with_eeg_l_1': False,
           'with_n_e_cv1': False,
           'with_n_e_cvl': False,
           'with_por_cv1': False,
           'with_por_cvl': False,
           'with_por_enc': False,
           'with_por_ate': False,
           'with_poi_cv1': False,
           'with_poi_cvl': False,
           'with_poi_enc': False,
           'with_poi_ate': False,
           'with_fea': True,
           }
    
    for key in turn_on: dic[key] = True
    
    return dic

# Aims at filtering the NaN and replace them with mean values
# arr refers to a 2D numpy array
def remove_out_with_mean(arr):
    
    col = np.unique(np.where(np.isnan(arr))[1])
    
    for idx in col:
        mea = np.nanmean(arr[:,idx])
        ind = np.where(np.isnan(arr[:,idx]))[0]
        arr[ind,idx] = mea
    
    col = np.unique(np.where(np.invert(np.isfinite(arr)))[1])
    
    for idx in col:
        tmp = arr[:,idx]
        mea = np.nanmean(tmp[np.where(np.isfinite(tmp))[0]])
        ind = np.where(np.invert(np.isfinite(tmp)))[0]
        arr[ind,idx] = mea
        
    return arr
