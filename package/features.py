# DINDIN Meryll
# Nov 19th, 2018
# Dreem Headband Sleep Phases Classification Challenge

try: from package.toolbox import *
except: from toolbox import *

# Computes the pairwise distances between EEGs
# signal refers to a 1D array

def frequency_features(signal, brain=False):

    res, f_s = [], int(len(signal)/30.0)

    # Basic features
    f,s = sg.periodogram(signal, fs=f_s)
    res.append(f[s.argmax()])
    res.append(np.max(s))
    res.append(np.sum(s))
    res.append(entropy(s))
    
    # Brain waves frequencies
    if brain:

        res.append(np.sum(s[np.where((f > 0.5) & (f < 3.0))[0]]))
        res.append(np.sum(s[np.where((f > 3.0) & (f < 8.0))[0]]))
        res.append(np.sum(s[np.where((f > 12.) & (f < 38.))[0]]))
        res.append(np.sum(s[np.where((f > 38.) & (f < 42.))[0]]))

        f,_,S = sg.spectrogram(signal, fs=f_s, return_onesided=True)
        res += list(f[S.argmax(axis=0)])
        psd = np.sum(S, axis=0)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

        f,_,Z = sg.stft(signal, fs=f_s, window='hamming', nperseg=int(5*f_s), noverlap=int(0.75*5*f_s))
        Z = np.abs(Z.T)
        res += list(f[Z.argmax(axis=1)])
        psd = np.sum(Z, axis=1)
        res.append(np.mean(psd))
        res.append(np.std(psd))
        res.append(entropy(psd))

    return res

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

def wavelet_features(val):

    res = []

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

# Compute specific entropy features
# val refers to a 1D array

def neural_entropy_features(val):

    warnings.simplefilter('ignore')

    res, f_s = [], int(len(val) / 30.0)
    
    arg = {'lyap_r': False, 'lyap_e': True, 'sampen': False, 'multiscale': False}
    tmp = neurokit.complexity(val, sampling_rate=f_s, **arg)
    for key in sorted(list(tmp.keys())): res.append(tmp[key])
    res.append(nolds.sampen(val))
    res.append(nolds.lyap_r(val))

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

# Defines the feature construction pipeline
# val refers to a 1D array

def stats_features(val):

    def nested_stats(signal):
        
        res = []

        nan = np.where(np.invert(np.isnan(signal)))[0]
        
        res.append(min(signal[nan]))
        res.append(max(signal[nan]))
        res.append(np.nanmean(signal[nan]))
        res.append(np.nanstd(signal[nan]))
        res.append(kurtosis(signal[nan]))
        res.append(skew(signal[nan]))
        res.append(entropy(signal[nan]))
        
        return res
    
    # Build the feature vector
    res = nested_stats(val)
    for per in [25, 50, 75]: res.append(np.percentile(val, per))

    # Common features
    tmp = seasonal_decompose(val, model='additive', freq=int(len(val)/30))
    res += nested_stats(tmp.trend)
    res.append(crossing_over(tmp.trend))
    res += ar_coefficients(tmp.trend)
    res += nested_stats(tmp.resid)
    
    return np.asarray(res)

# General computation of features

def compute_features(val, brain=False):

    res = list(stats_features(val))
    res += list(frequency_features(val, brain=brain))

    if brain:

        res += list(neural_entropy_features(val))
        res += list(wavelet_features(val))

    return np.asarray(res)

# Compute pairwise euclidean distance between eeg signals

def compute_distances(idx, h5_path='./dataset/train.h5'):
    
    with h5py.File(h5_path, 'r') as dtb:
        vec = np.asarray([dtb[key][idx] for key in ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']])
        
    return pairwise_distances(vec)[np.triu_indices(4, k=1)]
