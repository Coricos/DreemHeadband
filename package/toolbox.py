# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.imports import *

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

# Compute the fft components
# val refers to a 1D array
# n_components refers to the desired amounts of components
def compute_fft(val, n_components=50):

    return np.abs(np.fft.rfft(val))[:n_components]

# Compute features related to the chaos theory
# val refers to a 1D array
def compute_chaos(val):

    res = []
    res.append(nolds.sampen(val))
    res.append(nolds.dfa(val))
    res.append(nolds.hurst_rs(val))
    res.append(nolds.lyap_r(val))
    res += list(nolds.lyap_e(val))

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
        
        dta = np.round(val, 2)
        cnt = Counter()

        for ele in dta: cnt[ele] += 1

        pbs = [val / len(dta) for val in cnt.values()]
        pbs = [prd for prd in pbs if prd > 0.0]

        ent = 0.0
        for prd in pbs:
            ent -= prd * log(prd, 2.0)

        return ent

    res = []
    # Build the feature vector
    res.append(np.mean(val))
    res.append(np.std(val))
    res.append(min(val))
    res.append(max(val))
    # Frequential features
    fft = np.abs(np.fft.rfft(val))
    res.append(np.trapz(fft))
    for per in [25, 50, 75]: res.append(np.percentile(fft, per))
    f,s = sg.periodogram(val, fs=len(val)/30)
    idx = np.argsort(s)[::-1]
    res += list(f[idx][:10])
    res += list(s[idx][:10])
    c,_ = pywt.cwt(val, np.arange(1, 64), 'cmor', 30)
    coe = (np.square(np.abs(c))).mean(axis=0)
    res.append(np.trapz(coe))
    for per in [25, 50, 75]: res.append(np.percentile(coe, per))
    hil = sg.hilbert(val)
    i_p = np.unwrap(np.angle(hil))
    i_f = np.diff(i_p) / (2 * np.pi * 130)
    res.append(np.trapz(np.abs(hil)))
    res.append(max(np.abs(hil)))
    for per in [25, 50, 75]: res.append(np.percentile(np.abs(hil), per))
    res.append(np.trapz(i_f))
    res.append(max(i_f))
    res.append(min(i_f))
    res.append(np.std(i_f))
    for per in [25, 50, 75]: res.append(np.percentile(np.abs(i_f), per))
    # Statistical features
    res.append(kurtosis(val))
    res.append(skew(val))
    res.append(crossing_over(val))
    res.append(entropy(val))
    grd = np.gradient(val)
    res.append(np.mean(grd))
    res.append(np.std(grd))
    res.append(min(grd))
    res.append(max(grd))

    # Memory efficiency
    del grd, fft, f, s, c, coe, hil, i_p, i_f

    return np.asarray(res)

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

# Compute the wavelet spectrum of the signal
# vec refers to a 1D array
def compute_wavelet(vec):

    coe,_ = pywt.cwt(vec, np.arange(1, 128), 'cmor', 30)
    
    return (np.square(np.abs(coe))).mean(axis=0)
