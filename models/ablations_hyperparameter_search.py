#Random search over hyperparameters for each ablation feature set.

import os
import pickle, random
import numpy as np
import pandas as pd
from time import time
from LNP_model import pandas_classifier

DATA_DIR      = '../datasets/ablations'
PARAMS_DIR    = './params/ablations'
FEATURE_SETS  = ['bulk', 'mordred', 'sapt',
                 'bulk+mordred', 'bulk+sapt', 'sapt+mordred',
                 'combined']
MAX_EVALS     = 100
SEED          = 15

DROP_COLS = ['label', 'Payload', 'Lipomer', 'Cholesterol',
             'HelperLipid', 'PEGChain', 'PEG MW', 'diameter']

#Random search for hyperparameter optimization
def random_search(data, y, param_grid, constant_params, max_evals = 100):
    # Dataframe for results
    results = pd.DataFrame(columns = ['auc', 'accuracy', 'params', 'iteration'],
                                  index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters.update(constant_params)
        
        # Evaluate randomly selected hyperparameters
        df = pd.concat([data, pd.DataFrame(pd.Series(y, name='label'))], axis=1)
        importance_array = pd.DataFrame()
        verbose = False
        importance_array, auc, accuracy = pandas_classifier(df, 1, 5, importance_array, hyperparameters, verbose = False)
        
        #Add row to results df
        results.loc[i, :] = [auc, accuracy, hyperparameters, i]
        
    # Sort with best score on top
    results.sort_values('auc', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results

def run_feature_set(feature_set, param_grid, constant_params):
    print(f'\n=== ablation: {feature_set} ===')
    data_train = pd.read_csv(f'{DATA_DIR}/{feature_set}_train.csv')

    y_train = np.array(data_train['label'])
    X_train = data_train.drop(columns=[c for c in DROP_COLS if c in data_train.columns])

    np.random.seed(SEED)
    random.seed(SEED)

    t0 = time()
    results = random_search(X_train, y_train, param_grid, constant_params, MAX_EVALS)
    elapsed = (time() - t0) / 60
    print(f'  time elapsed: {elapsed:.1f} min')

    best_params   = results.loc[0, 'params']
    best_auc      = results.loc[0, 'auc']
    best_accuracy = results.loc[0, 'accuracy']
    print(f'  best AUC:      {best_auc:.4f}')
    print(f'  best Accuracy: {best_accuracy:.4f}')

    results.to_csv(f'{PARAMS_DIR}/RS_results_{feature_set}.csv', index=False)
    with open(f'{PARAMS_DIR}/hyperparameters_{feature_set}.pkl', 'wb') as f:
        pickle.dump(best_params, f, pickle.HIGHEST_PROTOCOL)

def main():
    os.makedirs(PARAMS_DIR, exist_ok=True)

    param_grid = {
        'eta':              [0.01, 0.05, 0.07, 0.1, 0.12, 0.14, 0.2],
        'max_depth':        list(range(3, 10)),
        'gamma':            [i / 10.0 for i in range(0, 5)],
        'alpha':            [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10],
        'lambda':           [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10],
        'subsample':        [i / 10.0 for i in range(5, 10)],
        'colsample_bytree': [i / 10.0 for i in range(5, 10)],
    }
    constant_params = {
        'objective':        'binary:logistic',
        'min_child_weight': 1,
        'verbosity':        0,
        'nthread':          6,
        'eval_metric':      'auc',
        'seed':             SEED,
        'scale_pos_weight': 3,
    }

    for fset in FEATURE_SETS:
        run_feature_set(fset, param_grid, constant_params)

if __name__ == '__main__':
    main()