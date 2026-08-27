#Baseline model comparison on the combined feature set.
#Logistic regression, naive Bayes, KNN, decision tree, random forest

import os
import random
import warnings

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings('ignore')

DATA_DIR    = '../datasets/'
RESULTS_DIR = './baseline_results'

FEATURE_SET = 'RNA'
K_FOLDS     = 5
N_BOOTSTRAP = 1000
SEED        = 15

DROP_COLS = ['label', 'Payload', 'Lipomer', 'Cholesterol',
             'HelperLipid', 'PEGChain', 'PEG MW', 'diameter']

METRIC_NAMES = ['ROC_AUC', 'PR_AUC', 'Accuracy', 'Balanced_Accuracy',
                'Sensitivity', 'Specificity', 'Precision', 'F1']

MODELS = {
    'logistic_regression': lambda: LogisticRegression(random_state=SEED, max_iter=5000),
    'decision_tree':       lambda: DecisionTreeClassifier(random_state=SEED),
    'random_forest':       lambda: RandomForestClassifier(random_state=SEED),
    'knn':                 lambda: KNeighborsClassifier(),
}

GRIDS = {
    'logistic_regression': {'C': [0.01, 0.1, 1, 10]},
    'decision_tree':       {'max_depth': [5, 10, 20, None],
                            'min_samples_leaf': [1, 5, 10]},
    'random_forest':       {'n_estimators': [100, 300, 500],
                            'max_depth': [10, 20, None],
                            'max_features': ['sqrt', 'log2']},
    'knn':                 {'n_neighbors': [3, 5, 15, 25],
                            'weights': ['uniform', 'distance']},
}

def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'ROC_AUC':           roc_auc_score(y_true, y_score),
        'PR_AUC':            average_precision_score(y_true, y_score),
        'Accuracy':          accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity':       tp / (tp + fn) if (tp + fn) else np.nan,
        'Specificity':       tn / (tn + fp) if (tn + fp) else np.nan,
        'Precision':         tp / (tp + fp) if (tp + fp) else np.nan,
        'F1':                f1_score(y_true, y_pred, zero_division=0),
    }

def bootstrap_metrics(y_true, y_score, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    point = compute_metrics(y_true, y_score)
    boots = {m: [] for m in METRIC_NAMES}
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    for _ in range(n):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        q = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        for k, v in compute_metrics(y_true[idx], y_score[idx]).items():
            boots[k].append(v)
    return {m: (point[m],
                float(np.percentile(boots[m], 2.5)),
                float(np.percentile(boots[m], 97.5))) for m in METRIC_NAMES}

def load_data():
    train = pd.read_csv(f'{DATA_DIR}/{FEATURE_SET}_train.csv')
    test  = pd.read_csv(f'{DATA_DIR}/{FEATURE_SET}_test.csv')
    y_train = np.array(train['label']).astype(int)
    y_test  = np.array(test['label']).astype(int)
    X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    X_test  = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    return X_train, y_train, X_test, y_test

def predict_score(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, 'decision_function'):
        s = clf.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    return clf.predict(X).astype(float)

def tune_and_fit(model_ctor, grid, X_train, y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(model_ctor(), grid, cv=kf, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

#5 fold cross validation, then final train and test
def tune_cv_and_test(name, model, X_train, y_train, X_test, y_test):
    grid = GRIDS[name]
    scaler = StandardScaler()
    Xtrain  = scaler.fit_transform(X_train)
    Xtest  = scaler.transform(X_test)

    # Grid search with 5fold CV
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(model(), grid, cv=kfold, scoring='roc_auc',
                       n_jobs=-1, refit=True)
    grid_search.fit(Xtrain, y_train)

    print(f'  best params: {grid_search.best_params_}')
    print(f'  best CV ROC-AUC: {grid_search.best_score_:.3f}')

    # Extract per-fold CV AUCs for the best config
    best_idx = grid_search.best_index_
    cv_aucs = [grid_search.cv_results_[f'split{i}_test_score'][best_idx]
               for i in range(K_FOLDS)]

    # Score on held-out test set with refit best model
    y_score = predict_score(grid_search.best_estimator_, Xtest)

    return cv_aucs, y_score, grid_search.best_params_

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    X_train, y_train, X_test, y_test = load_data()
    print(f'Loaded combined dataset: train={X_train.shape}, test={X_test.shape}')

    rows = []
    param_rows = []
    for name, ctor in MODELS.items():
        print(f'\n=== {name} ===')
        cv_aucs, y_score, best_params = tune_cv_and_test(
            name, ctor, X_train, y_train, X_test, y_test)
        print(f'  CV ROC-AUC per fold: {[round(a, 3) for a in cv_aucs]}')
        print(f'  CV ROC-AUC mean ± std: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}')

        metrics_ci = bootstrap_metrics(y_test, y_score)

        row = {'model': name,
               'CV_ROC_AUC_mean': float(np.mean(cv_aucs)),
               'CV_ROC_AUC_std':  float(np.std(cv_aucs))}
        for m, (point, lo, hi) in metrics_ci.items():
            row[m]              = point
            row[f'{m}_low95']   = lo
            row[f'{m}_high95']  = hi
        rows.append(row)

        param_rows.append({'model': name, **best_params})

    results = pd.DataFrame(rows)
    results.to_csv(f'{RESULTS_DIR}/baseline_metrics.csv', index=False)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(f'{RESULTS_DIR}/baseline_hyperparameters.csv', index=False)

    print('\n=== Baseline results ===')
    print(results.round(3).to_string(index=False))
    print('\n=== Best hyperparameters ===')
    print(params_df.to_string(index=False))


if __name__ == '__main__':
    main()