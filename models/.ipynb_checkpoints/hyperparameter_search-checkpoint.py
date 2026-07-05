"""Random search to produce the tuned hyperparameters used in the paper
"""

import pickle, random
import numpy as np
import pandas as pd
from time import time
from LNP_model import pandas_classifier

def random_search(data, y, param_grid, constant_params, max_evals = 100):
    """Random search for hyperparameter optimization"""
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
        importance_array, auc, accuracy = pandas_classifier(df, 1, 5, importance_array, hyperparameters, verbose)
        
        #Add row to results df
        results.loc[i, :] = [auc, accuracy, hyperparameters, i]
        
    # Sort with best score on top
    results.sort_values('auc', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 

def main():
    #Load data
    model_name = 'RNA' #'mRNA', 'siRNA' 

    data_train = pd.read_csv('../datasets/'+model_name+'_train.csv')
    data_test = pd.read_csv('../datasets/'+model_name+'_test.csv')

    #Drop Payload if not RNA model (RNA model contains both mRNA and siRNA nanoparticles)
    # data_train = data_train.drop(['Payload'],1)
    # data_test = data_test.drop(['Payload'],1)
        
    #Separate X and y
    y_train = np.array(data_train['label'])
    X_train = data_train.drop(['label','Lipomer', 'Cholesterol', 'HelperLipid', 'PEGChain', 'PEG MW', 'diameter'], 1)
    y_test = np.array(data_test['label'])
    X_test = data_test.drop(['label','Lipomer', 'Cholesterol', 'HelperLipid', 'PEGChain', 'PEG MW', 'diameter'], 1)

    #Hyperparameter Tuning - Random Search

    # Hyperparameter grid
    param_grid = {
        'eta': [0.01, 0.05, 0.07, 0.1, 0.12, 0.14, 0.2],
        'max_depth': list(range(3,10)),
        'gamma': [i/10.0 for i in range(0,5)],
        'alpha': [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10],
        'lambda': [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10],
        'subsample': [i/10.0 for i in range(5,10)],
        'colsample_bytree': [i/10.0 for i in range(5,10)],
    }

    constant_params = {
        'objective': 'binary:logistic', #'multi:softprob' multi:softmax'
        'min_child_weight': 1,
        'verbosity': 0,
        'nthread': 6,
        'eval_metric': 'auc',#'mlogloss'
        'seed': 15,
        'scale_pos_weight': 3, # sum(negative cases)/sum(positive cases)
    }
    
    np.random.seed(15)
    random.seed(15)
    time_1 = time()
    random_results = random_search(X_train, y_train, param_grid, constant_params, 100)
    time_2 = time()
    time_elapsed = (time_2 - time_1)/60
    print('Time Elapsed: ',time_elapsed,' minutes')

    # Get the best parameters
    random_search_params = random_results.loc[0, 'params']
    random_search_AUC = random_results.loc[0, 'auc']
    random_search_accuracy = random_results.loc[0, 'accuracy']

    print('Best Parameters: ',random_search_params)
    print('Best AUC: ',random_search_AUC)
    print('Best Accuracy: ',random_search_accuracy)

    #Store Results
    name = model_name
    random_results.to_csv('./params/RS_results_'+name+'.csv', index=False)
    with open('./params/hyperparameters'+name+'.pkl', 'wb') as f:
            pickle.dump(random_search_params, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()