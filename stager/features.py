# DINDIN Meryll
# Nov 19th, 2018
# Dreem Headband Sleep Phases Classification Challenge

try: from package.toolbox import *
except: from toolbox import *

# Computes the pairwise distances between EEGs
# signal refers to a 1D array

def frequency_features(signal, brain=False, sig_name=None):

    res, f_s = [], int(len(signal)/30.0)

    # Basic features
    f,s = sg.periodogram(signal, fs=f_s)
    res.append(f[s.argmax()])
    res.append(np.max(s))
    res.append(np.sum(s))
    res.append(entropy(s))

    if sig_name:

        nme = []
        nme.append('_'.join([sig_name, 'max_freq']))
        nme.append('_'.join([sig_name, 'max_spectrum']))
        nme.append('_'.join([sig_name, 'sum_spectrum']))
        nme.append('_'.join([sig_name, 'entropy_spectrum']))

    # Brain waves frequencies
    if brain:

        res.append(np.sum(s[np.where((f > 0.5) & (f < 3.0))[0]]))
        res.append(np.sum(s[np.where((f > 3.0) & (f < 8.0))[0]]))
        res.append(np.sum(s[np.where((f > 12.) & (f < 38.))[0]]))
        res.append(np.sum(s[np.where((f > 38.) & (f < 42.))[0]]))

        if sig_name:

            nme.append('_'.join([sig_name, 'sum_delta_waves']))
            nme.append('_'.join([sig_name, 'sum_theta_waves']))
            nme.append('_'.join([sig_name, 'sum_alpha_waves']))
            nme.append('_'.join([sig_name, 'sum_beta_waves']))

        f,_,S = sg.spectrogram(signal, fs=f_s, return_onesided=True)
        res += list(f[S.argmax(axis=0)])
        psd = np.sum(S, axis=0)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

        if sig_name:

            nme += ['_'.join([sig_name, 'max_freq_{}'.format(i)]) for i in range(len(f[S.argmax(axis=0)]))]
            nme.append('_'.join([sig_name, 'mean_psd']))
            nme.append('_'.join([sig_name, 'std_psd']))
            nme.append('_'.join([sig_name, 'entropy_psd']))

        f,_,Z = sg.stft(signal, fs=f_s, window='hamming', nperseg=int(5*f_s), noverlap=int(0.75*5*f_s))
        Z = np.abs(Z.T)
        res += list(f[Z.argmax(axis=1)])
        psd = np.sum(Z, axis=1)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

        if sig_name:

            nme += ['_'.join([sig_name, 'max_sliding_freq_{}'.format(i)]) for i in range(len(f[Z.argmax(axis=1)]))]
            nme.append('_'.join([sig_name, 'mean_sliding_psd']))
            nme.append('_'.join([sig_name, 'std_sliding_psd']))
            nme.append('_'.join([sig_name, 'entropy_sliding_psd']))

    if sig_name: return res, nme
    else: return res

# Parameters of the autoregressive models
# vec refers to a 1D array

def ar_coefficients(vec):

    nan = np.where(np.invert(np.isnan(vec)))[0]

    mod = AR(vec[nan])
    mod = mod.fit()
    
    return list(mod.params)

# Defines the amount of crossing-overs
# val refers to a 1D array

def crossing_over(val):
    
    nan = np.where(np.invert(np.isnan(val)))[0]
    
    sgn = np.sign(val[nan])
    sgn[sgn == 0] == -1

    return len(np.where(np.diff(sgn))[0])

# Defines the entropy of the signal
# val refers to a 1D array

def entropy(val):

    nan = np.where(np.invert(np.isnan(val)))[0]
    vec = val[nan]
    
    dta = np.round(vec, 5)
    cnt = Counter()

    for ele in dta: cnt[ele] += 1

    pbs = [vec / len(dta) for vec in cnt.values()]
    pbs = [prd for prd in pbs if prd > 0.0]

    ent = 0.0
    for prd in pbs:
        ent -= prd * log(prd, 2.0)

    return ent

# Defines the wavelet features
# val refers to a 1D array

def wavelet_features(val, sig_name=None):

    res = []

    cA_5, cD_5, cD_4, cD_3, _, _ = pywt.wavedec(val, 'db4', level=5)

    for sig in [cA_5, cD_5, cD_4, cD_3]:
        res += [np.min(sig), np.max(sig), np.sum(np.square(sig)), np.mean(sig), np.std(sig)]
    
    if sig_name:

        nme = []
        for key in ['A_5', 'D_5', 'D_4', 'D_3']:
            nme.append('_'.join([sig_name, 'min', key]))
            nme.append('_'.join([sig_name, 'max', key]))
            nme.append('_'.join([sig_name, 'sum', key]))
            nme.append('_'.join([sig_name, 'mean', key]))
            nme.append('_'.join([sig_name, 'std', key]))

    sgn = np.sign(val)
    sgn = np.split(sgn, np.where(np.diff(sgn) != 0)[0]+1)
    sgn = np.asarray([len(ele) for ele in sgn])
    res += [np.nanmean(sgn), np.std(sgn)]

    if sig_name:
        nme.append('_'.join([sig_name, 'mean_sign']))
        nme.append('_'.join([sig_name, 'std_sign']))
    
    sgn, ine = np.asarray([0] + list(sgn)), 0.0
    for idx in range(len(sgn)-1):
        ine += (sgn[idx+1] - sgn[idx])*np.trapz(np.abs(val[np.sum(sgn[:idx]):np.sum(sgn[:idx+1])]))
    res.append(ine)

    if sig_name: nme.append('_'.join([sig_name, 'sections_area']))

    if sig_name: return res, nme
    else: return res

# Compute specific entropy features
# val refers to a 1D array

def neural_entropy_features(val, sig_name=None):

    warnings.simplefilter('ignore')

    res, f_s = [], int(len(val) / 30.0)
    if sig_name: nme = []
    
    arg = {'lyap_r': False, 'lyap_e': True, 'sampen': False, 'multiscale': False}
    tmp = neurokit.complexity(val, sampling_rate=f_s, **arg)
    if sig_name: nme += ['_'.join([sig_name, ele]) for ele in sorted(list(tmp.keys()))]
    for key in sorted(list(tmp.keys())): res.append(tmp[key])
    res.append(nolds.sampen(val))
    res.append(nolds.lyap_r(val))
    if sig_name:
        nme.append('_'.join([sig_name, 'sampen']))
        nme.append('_'.join([sig_name, 'lyap_r']))

    dif = np.diff(val)
    dif = np.asarray([val[0]] + list(dif))
    num = len(val)
    m_2 = float(np.sum(dif ** 2)) / num
    t_p = np.sum(np.square(val))
    res.append(np.sqrt(m_2 / t_p))
    if sig_name: nme.append('_'.join([sig_name, 'Hjorth']))
    
    m_4 = 0.0
    for idx in range(1, len(dif)):
        m_4 += np.square(dif[idx] - dif[idx-1])
    m_4 = m_4 / num
    res.append(np.sqrt(m_4 * t_p / m_2 / m_2))
    if sig_name: nme.append('_'.join([sig_name, 'fractal']))

    if sig_name: return res, nme
    else: return res

# Defines the feature construction pipeline
# val refers to a 1D array

def stats_features(val, sig_name=None):

    def nested_stats(signal, sig_name=None):
        
        res = []

        nan = np.where(np.invert(np.isnan(signal)))[0]
        
        res.append(min(signal[nan]))
        res.append(max(signal[nan]))
        res.append(np.nanmean(signal[nan]))
        res.append(np.nanstd(signal[nan]))
        res.append(kurtosis(signal[nan]))
        res.append(skew(signal[nan]))
        res.append(entropy(signal[nan]))

        if sig_name:

            nme = []
            nme.append('_'.join([sig_name, 'min']))
            nme.append('_'.join([sig_name, 'max']))
            nme.append('_'.join([sig_name, 'mean']))
            nme.append('_'.join([sig_name, 'std']))
            nme.append('_'.join([sig_name, 'kurtosis']))
            nme.append('_'.join([sig_name, 'skew']))
            nme.append('_'.join([sig_name, 'entropy']))
        
        if sig_name: return res, nme
        else: return res
    
    # Build the feature vector
    if sig_name: res, nme = nested_stats(val, sig_name=sig_name)
    else: res = nested_stats(val)

    for per in [25, 50, 75]: 
        if sig_name: nme.append('_'.join([sig_name, 'per_{}'.format(per)]))
        res.append(np.percentile(val, per))

    # Common features

    dec = seasonal_decompose(val, model='additive', freq=int(len(val)/30))
    if sig_name: 
        tmp, lbl = nested_stats(dec.trend, sig_name='_'.join([sig_name, 'trend']))
        res += tmp
        nme += lbl
    else: res += nested_stats(dec.trend)
    
    if sig_name: nme.append('_'.join([sig_name, 'cross_over']))
    res.append(crossing_over(dec.trend))

    if sig_name:
        tmp = ar_coefficients(dec.trend)
        res += tmp
        nme += ['{}_AR_{}'.format(sig_name, i) for i in range(len(tmp))]
    else: res += ar_coefficients(dec.trend)

    if sig_name: 
        tmp, lbl = nested_stats(dec.resid, sig_name='_'.join([sig_name, 'resid']))
        res += tmp
        nme += lbl
    else: res += nested_stats(dec.resid)
    
    if sig_name: return res, nme
    else: return res

# General computation of features

def compute_features(val, brain=False, sig_name=None):

    if sig_name:
        res, nme = stats_features(val, sig_name=sig_name)
        tmp, lbl = frequency_features(val, brain=brain, sig_name=sig_name)
        res += tmp
        nme += lbl
    else:
        res = list(stats_features(val))
        res += list(frequency_features(val, brain=brain))

    if brain:

        if sig_name:
            tmp, lbl = neural_entropy_features(val, sig_name=sig_name)
            res += tmp
            nme += lbl
        else: res += list(neural_entropy_features(val))

        if sig_name:
            tmp, lbl = wavelet_features(val, sig_name=sig_name)
            res += tmp
            nme += lbl
        else: res += list(wavelet_features(val))

    if sig_name: return np.asarray(res), np.asarray(nme)
    else: return np.asarray(res)

# Compute pairwise euclidean distance between eeg signals

def compute_distances(idx, h5_path='./dataset/train.h5'):
    
    with h5py.File(h5_path, 'r') as dtb:
        vec = np.asarray([dtb[key][idx] for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']])
        
    return pairwise_distances(vec)[np.triu_indices(4, k=1)]

# Returns the list of feature labels

def give_name_to_features():
    
    lab, res = [], []

    for sig in ['po_r', 'po_ir', 'acc_x', 'acc_y', 'acc_z', 'norm_acc']:
        tmp = np.random.uniform(-10, 10, size=1500)
        ind, lbl = compute_features(tmp, brain=False, sig_name=sig)
        res += list(ind)
        lab += list(lbl)

    for sig in ['norm_eeg', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']:
        tmp = np.random.uniform(-10, 10, size=3750)
        ind, lbl = compute_features(tmp, brain=True, sig_name=sig)
        res += list(ind)
        lab += list(lbl)

    sig = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
    lab += list(np.asarray([['{}_to_{}'.format(i, j) for i in sig] for j in sig])[np.triu_indices(4, k=1)])

    for sig in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir', 'norm_acc', 'norm_eeg']:
        lab += ['_'.join([sig, 'pca_{}'.format(i)]) for i in range(5)]
        
    return lab