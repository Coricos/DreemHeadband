# Author:  Meryll Dindin
# Date:    02 March 2020
# Project: DreemEEG

from manager import *

if __name__ == '__main__':

    with h5py.File('raw-data/X_train.h5', 'r') as dtb: x_t = dtb.get('features')[()]
    y_t = pd.read_csv('raw-data/y_train.csv').set_index('id').values.ravel()
    with h5py.File('raw-data/X_test.h5', 'r') as dtb: x_e = dtb.get('features')[()]
    y_e = np.full(x_e.shape[0], None)

    with Pool(processes=cpu_count()) as pol: 
        r, c = reduce(lambda x,y: x*y, x_t.shape[:-1]), x_t.shape[-1]
        df_t = pol.map(stringify, x_t.reshape(r, c))
        r, c = reduce(lambda x,y: x*y, x_e.shape[:-1]), x_e.shape[-1]
        df_e = pol.map(stringify, x_e.reshape(r, c))
        
    times = pd.DataFrame(df_t + df_e, columns=['ts'])
    times.index.name = 'id'
    times.index = times.index.astype(int)

    mts_t = np.asarray(list(product(*[range(e) for e in x_t.shape[:-1]])))
    lab_t = np.hstack(([np.full(40*7, l) for l in y_t]))
    lab_t = np.vstack((np.full(len(lab_t), 'train'), lab_t)).T
    mts_e = np.asarray(list(product(*[range(e) for e in x_e.shape[:-1]])))
    lab_e = np.hstack(([np.full(40*7, l) for l in y_e]))
    lab_e = np.vstack((np.full(len(lab_e), 'test'), lab_e)).T
    metas = np.hstack((np.vstack((mts_t, mts_e)), np.vstack((lab_t, lab_e))))

    colms = ['d{}'.format(i) for i in range(len(x_t.shape)-1)] + ['origin', 'label']
    metas = pd.DataFrame(metas, columns=colms)
    metas.index.name = 'id'
    metas.index = metas.index.astype(int)

    sql = SqlManager(local_path='raw-data/storage.db')
    sql.populate('Metas', metas, None, 'Series', times, None)
    sql.featurize('Series', 'Features', 250)
