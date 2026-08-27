#Feature-importance robustness analysis for the LNP stability classifier.
#computing XGBoost feature importance, SHAP values, and permutation importance

import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score

from LNP_model import (classifier_train, classifier_test,
                       feature_selection, trainXGB, show_confusion_matrix)

warnings.filterwarnings('ignore')

DATA_DIR       = '../datasets'
PARAMS_DIR     = './params'
RESULTS_DIR    = './results'

PAYLOADS       = ['mRNA', 'siRNA', 'RNA']
K              = 5
N_PERM_REPEATS = 20
TOP_N          = 30
SEED           = 15

METHOD_LABELS = {
    'fscore':      'feature importance',
    'shap':        'SHAP values',
    'permutation': 'Permutation importance',
}


#Model training with importance calculated per fold
def pandas_classifier_with_importance(df, runXGB, K, importance_array,
                                      hyperparameters, all_cols, verbose=True):
    print('Performing ' + str(K) + '-fold cross validation')
    auc_fold = []
    acc_fold = []

    n_all = len(all_cols)
    fold_matrices = {
        'fscore':      np.zeros((K, n_all)),
        'shap':        np.zeros((K, n_all)),
        'permutation': np.zeros((K, n_all)),
    }

    for k in range(K):  # performing K fold validation
        print('Fold_num = ' + str(k))
        training_rows = [i for i in range(len(df)) if i%K!=k]
        datatrain = df.loc[training_rows]  # training
        testing_rows = [i for i in range(len(df)) if i%K==k]
        datatest = df.loc[testing_rows]    # taking every k'th example for test
        Xtrain = datatrain.iloc[:, 0:-1]
        ytrain = datatrain.iloc[:, -1]
        Xtest = datatest.iloc[:, 0:-1]
        ytest = datatest.iloc[:, -1]
        print('--------------------------------------------------------------')
        print('Calling the classifier to train')
        importance_array = pd.DataFrame()
        Xtrain_scaled, ytrain, scaler, pca, clf, index, importance_array, column_names = classifier_train(Xtrain, ytrain, runXGB, Xtest, ytest, 0, 1, importance_array, Xtrain.columns, hyperparameters, verbose)
        print('Feature selection using training feature importance')
        Xtest_scaled = scaler.transform(Xtest)
        imp_features, Xtrain_scaled, Xtest_scaled, column_names = feature_selection(Xtrain_scaled, Xtest_scaled, importance_array, column_names, verbose)
        print('Repeat training on filtered training data')
        importance_array = pd.DataFrame()
        clf, importance_array = trainXGB(Xtrain_scaled, ytrain, Xtest_scaled, ytest, importance_array, column_names, hyperparameters, verbose)
        print('Analysing the test predictions for fold num ', k)
        pred_array, auc, clf = classifier_test(Xtest_scaled, ytest, clf, index, scaler, 0, verbose)
        auc_fold.append(auc[0])
        print('test auc = '+str(auc[0]) )
        accuracy = accuracy_score(ytest.astype(int), np.round(pred_array))
        acc_fold.append(accuracy)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        if verbose:
            show_confusion_matrix(ytest, np.round(pred_array))

        sel_cols = np.array(column_names)

        # feature importance
        fscore_vec = np.zeros(len(sel_cols))
        for _, row in importance_array.iterrows():
            fscore_vec[int(str(row['feature']).replace('f', ''))] = row['fscore']

        # SHAP values
        shap_vals = shap.TreeExplainer(clf).shap_values(Xtest_scaled)
        shap_vec = np.abs(shap_vals).mean(axis=0)

        # Permutation importance
        perm_vec = np.zeros(len(sel_cols))
        rng = np.random.default_rng(SEED + k)
        for j in range(len(sel_cols)):
            drops = []
            for _ in range(N_PERM_REPEATS):
                X_shuf = Xtest_scaled.copy()
                X_shuf[:, j] = rng.permutation(X_shuf[:, j])
                shuf_pred, _, _ = classifier_test(X_shuf, ytest, clf, index,
                                                  scaler, 0, False)
                drops.append(accuracy -
                             accuracy_score(ytest.astype(int),
                                            np.round(shuf_pred)))
            perm_vec[j] = float(np.mean(drops))

        # Broadcast each per-fold vector into the full feature space
        for name, vec in [('fscore', fscore_vec),
                          ('shap', shap_vec),
                          ('permutation', perm_vec)]:
            for i, feat in enumerate(sel_cols):
                idx = np.where(all_cols == feat)[0][0]
                fold_matrices[name][k, idx] = vec[i]

        print('------------------------------------------------------------')

    if K != 0:
        print('************************************************************************')
        print(auc_fold)
        print('Average '+str(K)+' fold CV AUC= ', str(sum(np.array(auc_fold))/int(K)))
        print('Average '+str(K)+' fold CV Accuracy= ', str(sum(np.array(acc_fold))/int(K)))
        print('************************************************************************')

    return fold_matrices, sum(np.array(auc_fold))/int(K), sum(np.array(acc_fold))/int(K)

def summarize_and_plot(fold_matrix, all_cols, method, payload, top_n=TOP_N):
    #CSV and horizontal-bar SVG with mean ± std across folds.
    mean    = fold_matrix.mean(axis=0)
    std     = fold_matrix.std(axis=0)
    ci_low  = np.percentile(fold_matrix, 2.5,  axis=0)
    ci_high = np.percentile(fold_matrix, 97.5, axis=0)

    df = pd.DataFrame({
        'feature': all_cols,
        'mean':    mean,
        'std':     std,
        'ci_low':  ci_low,
        'ci_high': ci_high,
    })
    for k in range(fold_matrix.shape[0]):
        df[f'fold_{k}'] = fold_matrix[k]
    df = df.sort_values('mean', ascending=False).reset_index(drop=True)
    df.to_csv(f'{RESULTS_DIR}/{method}_importance_{payload}.csv', index=False)

    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(top['feature'], top['mean'], xerr=top['std'],
            color='steelblue', ecolor='dimgray', capsize=2)
    ax.set_xlabel(f'{METHOD_LABELS[method]}  (mean ± std across {K} CV folds)')
    ax.set_title(f'{payload}  |  top {top_n} features')
    fig.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/{method}_importance_{payload}.svg',
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_payload(payload):
    print(f'\n=== {payload} ===')
    data_train = pd.read_csv(f'{DATA_DIR}/{payload}_train.csv')
    data_train = data_train.drop(columns=['Payload'], errors='ignore')

    y = np.array(data_train['label']).astype(int)
    X = data_train.drop(columns=['label', 'Lipomer', 'Cholesterol',
                                 'HelperLipid', 'PEGChain', 'PEG MW',
                                 'diameter'])

    with open(f'{PARAMS_DIR}/hyperparameters{payload}.pkl', 'rb') as f:
        hyperparams = pickle.load(f)
    hyperparams['tree_method'] = 'exact'
    hyperparams['base_score']  = 0.5

    # pandas_classifier expects a df with the label as the last column
    df = pd.concat([X, pd.DataFrame({'label': y})], axis=1)
    all_cols = X.columns.to_numpy()

    fold_matrices, mean_auc, mean_acc = pandas_classifier_with_importance(
        df, 1, K, pd.DataFrame(), hyperparams, all_cols, verbose=False)

    for method, mat in fold_matrices.items():
        summarize_and_plot(mat, all_cols, method, payload)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    for payload in PAYLOADS:
        run_payload(payload)


if __name__ == '__main__':
    main()