# Author: DINDIN Meryll
# Date: 27/04/2018
# Company: Fujitsu

from hyperband.manager import *

SPACE = dict()

SPACE_ETS = {

    'n_estimators': hp.choice('ne', [
        'default',
        hp.quniform('ne_', 10, 1000, 1)
    ]),
    'criterion': hp.choice('c', [
        'default',
        hp.choice('c_', ('gini', 'entropy'))
    ]),
    'max_depth': hp.choice('md', [
        'default',
        hp.quniform('md_', 2, 10, 1)
    ]),
    'bootstrap': hp.choice('bt', [
        'default',
        hp.choice('bt_', (True, False))
    ]),
    'max_features': hp.choice('mf', [
        'default',
        hp.choice('mf_', ('sqrt', 'log2', None))
    ]),
    'min_samples_split': hp.choice('mss', [
        'default',
        hp.quniform('mss_', 2, 20, 1)
    ]),
    'min_samples_leaf': hp.choice('msl', [
        'default',
        hp.quniform('msl_', 1, 10, 1)
    ])}

SPACE['ETS'] = SPACE_ETS

SPACE_GBT = {

    'n_estimators': hp.choice('ne', [
        'default',
        hp.quniform('ne_', 10, 1000, 1)
    ]),
    'learning_rate': hp.choice('lr', [
        'default',
        hp.uniform('lr_', 0.001, 0.3)
    ]),
    'subsample': hp.choice('ss', [
        'default',
        hp.uniform('ss_', 0.5, 1.0)
    ]),
    'loss': hp.choice('ls', [
        'default',
        hp.choice('ls_', ('ls', 'lad', 'huber', 'quantile'))
    ]),
    'criterion': hp.choice('c', [
        'default',
        hp.choice('c_', ('friedman_mse', 'mse', 'mae'))
    ]),
    'max_depth': hp.choice('md', [
        'default',
        hp.quniform('md_', 2, 10, 1)
    ]),
    'max_features': hp.choice('mf', [
        'default',
        hp.choice('mf_', ('sqrt', 'log2', None))
    ]),
    'min_samples_split': hp.choice('mss', [
        'default',
        hp.quniform('mss_', 2, 20, 1)
    ]),
    'min_samples_leaf': hp.choice('msl', [
        'default',
        hp.quniform('msl_', 1, 10, 1)
    ])}

SPACE['GBT'] = SPACE_GBT

SPACE_LGB = {

    'n_estimators': hp.choice('ne', [
        'default',
        hp.quniform('ne_', 10, 1000, 1)
    ]),
    'learning_rate': hp.choice('lr', [
        'default',
        hp.uniform('lr_', 0.001, 0.3)
    ]),
    'max_depth': hp.choice('md', [
        'default',
        hp.quniform('md_', 2, 10, 1)
    ]),
    'num_leaves': hp.choice('nl', [
        'default',
        hp.quniform('nl_', 2, 50, 1)
    ]),
    'min_child_weight': hp.choice('mcw', [
        'default',
        hp.quniform('mcw_', 1, 10, 1)
    ]),
    'min_child_samples': hp.choice('mcs', [
        'default',
        hp.quniform('mcs_', 10, 30, 1)
    ]),
    'subsample': hp.choice('ss', [
        'default',
        hp.uniform('ss_', 0.5, 1.0)
    ]),
    'colsample_bytree': hp.choice('cbt', [
        'default',
        hp.uniform('cbt_', 0.5, 1.0)
    ]),
    'reg_alpha': hp.choice('ra', [
        'default',
        hp.loguniform('ra_', log(1e-10), log(1))
    ]),
    'reg_lambda': hp.choice('rl', [
        'default',
        hp.uniform('rl_', 0.1, 10)
    ]),}

SPACE['LGB'] = SPACE_LGB

SPACE_RFS = {

    'n_estimators': hp.choice('ne', [
        'default',
        hp.quniform('ne_', 10, 1000, 1)
    ]),
    'criterion': hp.choice('c', [
        'default',
        hp.choice('c_', ('gini', 'entropy'))
    ]),
    'max_depth': hp.choice('md', [
        'default',
        hp.quniform('md_', 2, 10, 1)
    ]),
    'bootstrap': hp.choice('bt', [
        'default',
        hp.choice('bt_', (True, False))
    ]),
    'max_features': hp.choice('mf', [
        'default',
        hp.choice('mf_', ('sqrt', 'log2', None))
    ]),
    'min_samples_split': hp.choice('mss', [
        'default',
        hp.quniform('mss_', 2, 20, 1)
    ]),
    'min_samples_leaf': hp.choice('msl', [
        'default',
        hp.quniform('msl_', 1, 10, 1)
    ])}

SPACE['RFS'] = SPACE_RFS

SPACE_XGB = {

    'n_estimators': hp.choice('ne', [
        'default',
        hp.quniform('ne_', 10, 1000, 1)
    ]),
    'learning_rate': hp.choice('lr', [
        'default',
        hp.uniform('lr_', 0.001, 0.3)
    ]),
    'max_depth': hp.choice('md', [
        'default',
        hp.quniform('md_', 2, 10, 1)
    ]),
    'min_child_weight': hp.choice('mcw', [
        'default',
        hp.quniform('mcw_', 1, 10, 1)
    ]),
    'subsample': hp.choice('ss', [
        'default',
        hp.uniform('ss_', 0.5, 1.0)
    ]),
    'colsample_bytree': hp.choice('cbt', [
        'default',
        hp.uniform('cbt_', 0.5, 1.0)
    ]),
    'colsample_bylevel': hp.choice('cbl', [
        'default',
        hp.uniform('cbl_', 0.5, 1.0)
    ]), 
    'gamma': hp.choice('g', [
        'default',
        hp.uniform('g_', 0, 1)
    ]),
    'reg_alpha': hp.choice('ra', [
        'default',
        hp.loguniform('ra_', log(1e-10), log(1))
    ]),
    'reg_lambda': hp.choice('rl', [
        'default',
        hp.uniform('rl_', 0.1, 10)
    ]),
    'base_score': hp.choice('bs', [
        'default',
        hp.uniform('bs_', 0.1, 0.9)
    ]),
    'scale_pos_weight': hp.choice('spw', [
        'default',
        hp.uniform('spw', 0.1, 10)
    ])}

SPACE['XGB'] = SPACE_XGB

SPACE_SGD = {
    
    'loss': hp.choice('ls', [
        'default',
        hp.choice('ls_', ('squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'))
    ]),    
    'epsilon': hp.choice('ep', [
        'default', 
        hp.uniform('ep_', 0.001, 0.2)
    ]),
    'penalty': hp.choice('pe', [
        'default',
        hp.choice('pe_', ('none', 'l1', 'l2', 'elasticnet'))
    ]),
    'alpha': hp.choice('al', [
        'default', 
        hp.loguniform('al_', np.log(1e-10), np.log(1))
    ]),
    'l1_ratio': hp.choice('l1', [
        'default',
        hp.uniform('l1_', 0, 1)
    ]),
    'learning_rate': hp.choice('lr', [
        'default', 
        hp.choice('lr_', ('constant', 'optimal', 'invscaling'))
    ]),
    
    'eta0': hp.loguniform('et_', np.log(1e-10), np.log(1e-1)),

    'power_t': hp.choice('pt', [
        'default', 
        hp.uniform('pt_', 0.5, 0.99)
    ]),

    'fit_intercept': hp.choice('i', (True, False)),

    'shuffle': hp.choice('sh', (True, False))}

SPACE['SGD'] = SPACE_SGD

# Defines the relative parameters
def get_params(key):

    params = sample(SPACE[key])
    params = {k: v for k, v in params.items() if v is not 'default'}

    return handle_integers(params)

# Launch the discretisation of possibilities
def try_params(params, key, data):

    # Defines the new model
    if key == 'RFS':
        mod = RandomForestClassifier(**params)
    if key == 'GBT':
        mod = GradientBoostingClassifier(**params)
    if key == 'LGB':
        mod = lgb.LGBMClassifier(n_jobs=1, verbose=-1, objective='multiclass', **params)
    if key == 'ETS':
        mod = ExtraTreesClassifier(**params)
    if key == 'XGB':
        mod = xgb.XGBClassifier(**params)
    if key == 'SGD':
        mod = SGDClassifier(**params)
    
    return train_and_eval_classifier(mod, data)
