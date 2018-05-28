# Author: DINDIN Meryll
# Date: 27/04/2018
# Company: Fujitsu

from package.imports import *

# Handle floats which should be integers
# params refers to a set of parameters
def handle_integers(params):

    new_params = {}

    for k, v in params.items():

        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    
    return new_params

# Hyperband for regressors
# data is a dictionnary whose keys points towards the data used by the models
# clf refers to a regression model
def train_and_eval_regressor(clf, data):
    
    x_train, y_train = data['x_train'], data['y_train']
    x_valid, y_valid = data['x_valid'], data['y_valid']
    
    if 'w_train' in data.keys():
        clf.fit(x_train, y_train, sample_weight=data['w_train'])
    else:
        clf.fit(x_train, y_train)

    prd = clf.predict(x_train)

    if 'w_valid' in data.keys(): 
        mse = MSE(y_train, prd, sample_weight=data['w_train'])
        mae = MAE(y_train, prd, sample_weight=data['w_train'])
    else:
        mse = MSE(y_train, prd)
        mae = MAE(y_train, prd)

    print('\n# training | MSE: {:.10f}, MAE: {:.10f}'.format(mse, mae))

    prd = clf.predict(x_valid)

    if 'w_valid' in data.keys(): 
        mse = MSE(y_valid, prd, sample_weight=data['w_valid'])
        mae = MAE(y_valid, prd, sample_weight=data['w_valid'])
    else:
        mse = MSE(y_valid, prd)
        mae = MAE(y_valid, prd)

    print('# testing  | MSE: {:.10f}, MAE: {:.10f}'.format(mse, mae))
    
    return {'loss': mse, 'rmse': np.sqrt(mse), 'mae': mae}

# Hyperband for classifiers
# data is a dictionnary whose keys points towards the data used by the models
# clf refers to a classification model
def train_and_eval_classifier(clf, data):
    
    x_train, y_train = data['x_train'], data['y_train']
    x_valid, y_valid = data['x_valid'], data['y_valid'] 
    
    if 'w_train' in data.keys():
        clf.fit(x_train, y_train, sample_weight=data['w_train'])
    else:
        clf.fit(x_train, y_train)

    prd = clf.predict(x_train)

    if 'w_train' in data.keys():
        f1s = f1_score(y_train, prd, sample_weight=data['w_train'], average='weighted')
        acc = accuracy_score(y_train, prd, sample_weight=data['w_train'])
    else:
        f1s = f1_score(y_train, prd, average='weighted')
        acc = accuracy_score(y_train, prd)

    print('# Training | f1-score: {:.2%}, accuracy: {:.2%}'.format(f1s, acc))

    prd = clf.predict(x_valid)

    if 'w_valid' in data.keys():
        f1s = f1_score(y_valid, prd, sample_weight=data['w_valid'], average='weighted')
        acc = accuracy_score(y_valid, prd, sample_weight=data['w_valid'])
    else:
        f1s = f1_score(y_valid, prd, average='weighted')
        acc = accuracy_score(y_valid, prd)

    print('# Testing  | f1-score: {:.2%}, accuracy: {:.2%}'.format(f1s, acc))
    
    return {'f1_score': -f1s, 'acc': -acc}
