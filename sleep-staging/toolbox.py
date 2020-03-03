# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

try: from package.topology import *
except: from topology import *

# Defines a function to rename the datasets for clearer management
# storage refers to where to pick the dataset
def rename(storage='./dataset'):

    train_pth = '{}/train.h5'.format(storage)
    valid_pth = '{}/valid.h5'.format(storage)

    for pth in [train_pth, valid_pth]:

        with h5py.File(pth, 'a') as dtb:

            try:
                dtb['acc_x'] = dtb['accelerometer_x']
                del dtb['accelerometer_x']
            except: pass
            
            try: 
                dtb['acc_y'] = dtb['accelerometer_y']
                del dtb['accelerometer_y']
            except: pass

            try:
                dtb['acc_z'] = dtb['accelerometer_z']
                del dtb['accelerometer_z']
            except: pass

# Display a specific 30s sample
# idx refers to the index of the sample
def display(idx, storage='./dataset/valid.h5'):

    with h5py.File(storage, 'r') as dtb:
        # Load the signals
        a_x = dtb['accelerometer_x'][idx,:]
        a_y = dtb['accelerometer_y'][idx,:]
        a_z = dtb['accelerometer_z'][idx,:]
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
# size refers to the ending length
# log refers whether to apply a logarithmic scale on the signal
def resize_time_serie(val, size=400, log=False):

    if log:
        tmp = np.zeros(len(val))
        idx = np.where(val > 0)[0]
        tmp[idx] = np.log(1 + val[idx])
        idx = np.where(val < 0)[0]
        tmp[idx] = - np.log(1 + np.abs(val[idx]))
        tmp = interpolate(tmp, size=size)
    else:
        tmp = interpolate(val, size=size)
    
    return tmp

# Defines a vector reduction through interpolation
# val refers to a 1D array
# size refers to the desired size
def interpolate(val, size=1000):

    x = np.linspace(0, size, num=len(val), endpoint=True)
    o = np.linspace(0, size, num=size, endpoint=True)

    if len(val) < size:
        f = interp1d(x, val, kind='linear', fill_value='extrapolate')
        return f(o)

    if len(val) == size:
        return val

    if len(val) > size:
        f = interp1d(x, val, kind='quadratic')
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

# Multiprocessed way of computing the limits of a persistent diagrams
# vec refers to a 1D numpy array
def persistent_limits(vec):
    
    lvl = Levels(vec)
    u,d = lvl.get_persistence()
    
    return np.asarray([min(u[:,0]), max(u[:,1]), min(d[:,0]), max(d[:,1])])

# Compute the Betti curves
# vec refers to a 1D array
def compute_betti_curves(vec, mnu, mxu, mnd, mxd):

    fil = Levels(vec)
    try: v,w =  lvl.betti_curves(mnu, mxu, mnd, mxd, num_points=100)
    except: v,w = np.zeros(100), np.zeros(100)
    del fil
    
    return np.vstack((v,w))

# Compute the landscapes
# vec refers to a 1D array
def compute_landscapes(vec, mnu, mxu, mnd, mxd):

    fil = Levels(vec)
    try: p,q = fil.landscapes(mnu, mxu, mnd, mxd, num_points=100)
    except: p,q = np.zeros((10,100)), np.zeros((10,100))
    del fil
    
    return np.vstack((p,q))

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
           'with_fea': False,
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

# Returns the argmax of the most predicted value
# val refers to a 2D numpy array
def correlate(val, wei): 

    return np.argmax(np.sum(val/wei.reshape(wei.shape[0],1), axis=0))

# Defines an aggregater for ensemble learning
# storage refers where to pick the results files
# graph is a boolean for correlation
# out refers to a specific serialization path
def aggregate(storage='./results', graph=False, out=None):

    # Load all the existing validation results
    res, lst = [], glob.glob('{}/test_*.csv'.format(storage))
    for ele in lst: res.append(pd.read_csv(ele, sep=';', header=0, index_col=0))
    res = pd.concat(res, axis=1)
    res.columns = ['RES_{}'.format(k) for k in range(len(lst))]
    # Compute their respective correlation
    cor = res.corr()

    if graph:

        plt.figure(figsize=(18, 8))
        sns.heatmap(cor, cmap='BuGn', annot=True, cbar=False)
        plt.show()

    # Deal with the correlation for diversity
    wei = np.mean(cor, axis=0).values.ravel()
    res = res.values
    res = np_utils.to_categorical(res.ravel(), num_classes=5).reshape(res.shape[0], res.shape[1], 5)
    # Computes the respective scores
    pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    res = np.asarray(pol.map(partial(correlate, wei=wei), res))
    pol.close()
    pol.join()

    # Serialization of the result
    idx = np.arange(43830, 64422)
    res = np.hstack((idx.reshape(-1,1), res.reshape(-1,1)))
    # Creates the relative dataframe
    res = pd.DataFrame(res, columns=['id', 'label'])

    # Write to csv
    if out is None: out = '{}/aggr_{}.csv'.format(storage, int(time.time()))
    res.to_csv(out, index=False, header=True, sep=';')

# Multiprocessed bootstraping of time series
# vec refers to a 1D numpy array
# num refers to the amount of bootstraped elements
def bootstrap_sample(vec, num):
    
    res = [vec]
    bts = CircularBlockBootstrap(vec.shape[0], vec)
    for new in bts.bootstrap(num): res.append(new[0][0])
        
    return np.asarray(res)

# Visualization function for independent Conv1D channels
# channels refers to which channels to visualize
def independent_cv1(channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):
    
    for key in channels:

        with open('./models/HIS_CV1_{}.history'.format(key), 'rb') as raw: his = pickle.load(raw)

        plt.figure(figsize=(18,5))
        plt.suptitle('Training History | Independent Channel CV1 | {}'.format(key), y=1.05)
        idx = np.arange(len(his['output_loss']))
        plt.subplot(2,2,1)
        plt.plot(idx, his['output_loss'], label='Output Loss', c='orange')
        plt.scatter(idx, his['val_output_loss'], label='Val Output Loss', c='black', marker='d', s=5)
        plt.legend(loc='best')
        plt.grid()
        plt.subplot(2,2,2)
        plt.plot(idx, his['decode_loss'], label='Decode Loss', c='orange')
        plt.scatter(idx, his['val_decode_loss'], label='Val Decode Loss', c='black', marker='d', s=5)
        plt.legend(loc='best')
        plt.grid()
        plt.subplot(2,2,3)
        plt.plot(idx, his['output_acc'], label='Output Acc', c='orange')
        plt.scatter(idx, his['val_output_acc'], label='Val Output Acc', c='black', marker='d', s=5)
        plt.legend(loc='best')
        plt.grid()
        plt.subplot(2,2,4)
        plt.plot(idx, his['decode_mean_absolute_error'], label='Decode MAE', c='orange')
        plt.scatter(idx, his['val_decode_mean_absolute_error'], label='Val MAE', c='black', marker='d', s=5)
        plt.legend(loc='best')
        plt.grid()
        plt.tight_layout()
        plt.show()

# Visualization function for independent autoencoders
# channels refers to which channels to visualize
def independent_ate(channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):
    
    for key in channels:

        with open('./models/HIS_ATE_{}.history'.format(key), 'rb') as raw: his = pickle.load(raw)

        plt.figure(figsize=(18,3))
        plt.suptitle('Training History | Independent Channel ATE | {}'.format(key), y=1.05)
        idx = np.arange(len(his['loss']))
        plt.subplot(1,2,1)
        plt.plot(idx, his['loss'], label='Loss', c='orange')
        plt.legend(loc='best')
        plt.grid()
        plt.subplot(1,2,2)
        plt.plot(idx, his['mean_absolute_error'], label='Mean Absolute Error', c='lightblue')
        plt.legend(loc='best')
        plt.grid()
        plt.tight_layout()
        plt.show()

# Extract outliers from specific distribution
# ele refers to an array of values
# threshold refers to how far from the median distribution we can go
def outlier_from_median(ele, threshold):
        
    val = np.abs(ele - np.median(ele))
    val = val / np.median(val) if np.median(val) else 0.0
    
    return np.where(val > threshold)[0]

# Obtains the importances of a specific model given a specific level
# lvl refers to a level of noise
# model refers to a given model
def get_importances(lvl, model):
    
    # Get rid of obvious outiers
    lab = pd.read_csv('./dataset/label.csv', sep=';', index_col=0)
    msk = np.load('./models/level_{}/row_mask.npy'.format(lvl))
    x_t = np.load('./dataset/fea_train.npy')[msk[:len(lab)]]
    x_v = np.load('./dataset/fea_valid.npy')[msk[len(lab):]]

    # Preprocessing
    vtf = VarianceThreshold()
    vtf.fit(np.vstack((x_t, x_v)))

    # Restrict to a subset of features
    use = ['norm_acc', 'norm_eeg', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
    fea = np.asarray(joblib.load('./dataset/features.jb'))[vtf.get_support()]
    msk = np.zeros(len(fea), dtype=bool)
    idx = [i for i in range(len(fea)) if fea[i].startswith(tuple(use))]
    msk[np.asarray(idx)] = True
    fea = fea[msk]
    
    warnings.simplefilter('ignore')
    # Retrieve the importances
    clf = [joblib.load('./models/level_{}/cv{}_mod_{}.jb'.format(lvl, i, model)) for i in range(5)]
    imp = np.sum([ele.feature_importances_ for ele in clf], axis=0)
    imp = imp / np.sum(imp)
    
    return imp, fea

# Smoothing method
# y refers to the input vector
# window_size refers to the averaging section
def savitzky_golay(y, window_size, order=2, deriv=0, rate=1):

    from math import factorial

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # Precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')

# Rebuild the whole probabilities for a specific level
# lvl refers to a level of noise
def get_prediction_from_level(lvl):
    
    warnings.simplefilter('ignore')

    # Get rid of obvious outiers
    lab = pd.read_csv('./dataset/label.csv', sep=';', index_col=0)
    msk = np.load('./models/level_{}/row_mask.npy'.format(lvl))
    x_t = np.load('./dataset/fea_train.npy')[msk[:len(lab)]]
    y_t = lab.values.ravel()[msk[:len(lab)]]
    x_v = np.load('./dataset/fea_valid.npy')[msk[len(lab):]]

    # Preprocessing
    vtf = VarianceThreshold()
    vtf.fit(np.vstack((x_t, x_v)))
    x_t = vtf.transform(x_t)
    x_v = vtf.transform(x_v)
    mms = MinMaxScaler()
    sts = StandardScaler(with_std=False)
    pip = Pipeline([('mms', mms), ('sts', sts)])
    pip.fit(np.vstack((x_t, x_v)))
    x_t = pip.transform(x_t)
    x_v = pip.transform(x_v)
    del x_v, mms, sts, pip

    # Restrict to a subset of features
    use = ['norm_acc', 'norm_eeg', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
    fea = np.asarray(joblib.load('./dataset/features.jb'))[vtf.get_support()]
    msk = np.zeros(len(fea), dtype=bool)
    idx = [i for i in range(len(fea)) if fea[i].startswith(tuple(use))]
    msk[np.asarray(idx)] = True
    x_t = x_t[:,msk]
    del use, fea, msk, idx, vtf

    # Look for missing predictions
    prb = np.load('./models/level_{}/prb_t_LGB.npy'.format(lvl))
    kap = np.round(kappa_score(y_t, np.argmax(prb, axis=1)), 4)
    mis = np.where(np.sum(prb, axis=1) == 0)[0]
    clf = [joblib.load('./models/level_{}/cv{}_mod_LGB.jb'.format(lvl, i)) for i in range(5)]
    prd = np.asarray([ele.predict_proba(x_t[mis]) for ele in clf])
    prd = np.sum(prd, axis=0) / 5
    prb[mis] = prd
    fin = np.round(kappa_score(y_t, np.argmax(prb, axis=1)), 4)

    return prb, [lvl, np.round(len(prb) / len(lab), 3), len(mis), kap, fin]

# Transform a dataframe into an image
def dtf_to_img(dtf, row_height=0.6, font_size=11, ax=None):

    # Basic needed attributes
    header_color, row_colors, edge_color = '#40466e', ['#f1f1f2', 'w'], 'w'
    bbox, header_columns = [0, 0, 1, 1], 0

    if ax is None:
        size = (18, dtf.shape[0]*row_height)
        fig, ax = plt.subplots(figsize=(size))
        ax.axis('off')

    mpl_table = ax.table(cellText=dtf.values, bbox=bbox, colLabels=dtf.columns, cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])

    return ax
