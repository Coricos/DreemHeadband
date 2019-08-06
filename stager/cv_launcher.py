# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse, warnings

from package.detection import *
from optiML import CrossClassification
from scipy.stats import pearsonr

# Defines a split generator

def split_generator(folds=10):

    lab = pd.read_csv('./dataset/label.csv', sep=';', index_col=0)
    msk = np.load('./models/row_mask.npy')[:len(lab)]
    pro = Profiles(lab)
    _,i = pro.build_profiles()

    kfl = KFold(n_splits=folds, shuffle=True)
    for _, ele in kfl.split(np.arange(len(i))):
        val = np.zeros(len(lab), dtype=bool)
        for idx in ele: val[i[idx][0]:i[idx][1]] = True
        yield np.where(np.invert(val)[msk])[0], np.where(val[msk])[0]

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-m', '--model', help='Type of model to use', type=str, default='LGB')
    prs.add_argument('-f', '--folds', help='Number of folds for cross-validation', type=int, default=5)
    prs.add_argument('-t', '--threads', help='Amount of threads', type=int, default=multiprocessing.cpu_count())
    prs.add_argument('-i', '--max_iter', help='Maximum iterations', type=int, default=100)
    prs.add_argument('-s', '--slurm', help='Whether to use slurm', type=bool, default=False)
    # Parse the arguments
    prs = prs.parse_args()

    # Get rid of obvious outiers
    lab = pd.read_csv('./dataset/label.csv', sep=';', index_col=0)
    msk = np.load('./models/row_mask.npy')
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

    # Restrict to a subset of features
    use = ['norm_acc', 'norm_eeg', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
    fea = np.asarray(joblib.load('./dataset/features.jb'))[vtf.get_support()]
    msk = np.zeros(len(fea), dtype=bool)
    idx = [i for i in range(len(fea)) if fea[i].startswith(tuple(use))]
    msk[np.asarray(idx)] = True
    x_t = x_t[:,msk]
    x_v = x_v[:,msk]

    # Launch the model optimization
    clf = CrossClassification(x_t, x_v, y_t, slurm=prs.slurm, threads=prs.threads)
    clf.launch(split_generator(folds=prs.folds), metric='kap', model=prs.model, max_iter=prs.max_iter)
