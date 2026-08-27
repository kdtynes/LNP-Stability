#Ablations to compare models with different subsets of bulk, sapt, and mordred features

import ast
import itertools
import operator
import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, precision_recall_curve, 
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

DATA_DIR       = '../datasets/ablations'
PARAMS_DIR     = './params'
RESULTS_DIR    = './ablation_results'

FEATURE_SETS   = ['bulk', 'mordred', 'sapt',
                  'bulk+mordred', 'bulk+sapt', 'sapt+mordred',
                  'combined']

HYPERPARAM_PAYLOAD = 'RNA'   # use the combined/mixed hyperparams for all ablations
N_BOOTSTRAP        = 1000
SEED               = 15

METRIC_NAMES = ['ROC_AUC', 'PR_AUC', 'Accuracy', 'Balanced_Accuracy',
                'Sensitivity', 'Specificity', 'Precision', 'F1']


#This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize = (5,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.array(cm)
        cm = np.around(cm/cm.sum(axis=1)[:, None]*100).astype('int')
        print("Percentage confusion matrix")
        print(cm.sum(axis=1))
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)

    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    return roc_auc, tpr[i], fpr[i]

def trainXGB(Xtrain, ytrain, Xtest, ytest, importance_array, column_names, hyperparameters, verbose):
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest)
    print('Setting XGB params')
    evallist  = [(dtest,'test'), (dtrain,'train')]
    num_round = 220
    print('training the XGB classifier')
    bst = xgb.train(hyperparameters, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=False)
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df1 = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df1['fscore'] = df1['fscore'] / df1['fscore'].sum()

    print('check size of col names: ',len(column_names))
    df1['feature_names'] = pd.Series([column_names[int(f[0].replace("f", ""))] for f in importance])

    importance_array = df1.copy()
    df1 = df1.nlargest(30, 'fscore')
    if verbose:
        df1.plot()
        df1.sort_values(by='fscore',ascending=True).plot(kind='barh', x='feature_names', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance (Top 30)')
        plt.xlabel('Relative importance')
        plt.gcf().savefig('feature_importance_xgb.svg')
        plt.show()

    return bst, importance_array


def classifier_train(X, y, runXGB, Xtest, ytest, pca_comp, smote, importance_array, column_names, hyperparameters, verbose):
    print('Normalising the input data...')
    scaler = StandardScaler()
    scaler.fit(X)
    scaledX = scaler.transform(X)
    if smote == 1:
        X_resampled, y_resampled = SMOTE().fit_resample(np.array(scaledX), y.astype(int))
        scaledX = X_resampled
        y = y_resampled
    if pca_comp != 0:
        pca = PCA(n_components = pca_comp)
        pca.fit(scaledX)
        print(pca.explained_variance_ratio_ * 100)
        pca_scaledX = pca.transform(scaledX)
    else:
        pca_scaledX = scaledX
        pca = 0

    if runXGB == 1:
        print('Running the XGB classifier')
        clf, importance_array = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest, importance_array, column_names, hyperparameters, verbose)
        index = 1
    return pca_scaledX, y, scaler, pca, clf, index, importance_array, column_names


def classifier_test(scaledX, y, clf, index, scaler, pca, verbose):
    pca_scaledX = scaledX
    # if pca != 0:
    #     pca_scaledX = pca.transform(scaledX)
    if index==1:
        pca_scaledXG = xgb.DMatrix(pca_scaledX, label=y)
        pred_array = clf.predict(pca_scaledXG)
        scores = pred_array
    auc = get_auc(y, scores)
    return pred_array, auc, clf


def feature_selection(Xtrain, Xtest, importance_array, column_names, verbose):
    imp_vals = importance_array
    imp_vals['feature'] = imp_vals['feature'].map(lambda k: k.replace("f",""))
    imp_vals['feature'] = imp_vals['feature'].astype(int)
    imp_vals['fscore'] = imp_vals['fscore'].astype(float)
    imp_vals = imp_vals.sort_values('feature')
    if verbose:
        print('---importance values---')
        print('max:',np.max(imp_vals.iloc[:,1]))
        print('min:',np.min(imp_vals.iloc[:,1]))
        print('mean:',np.mean(imp_vals.iloc[:,1]))
    threshold = 0
    imp_vals = imp_vals.loc[imp_vals['fscore'] > threshold]
    filt = imp_vals.to_numpy()[:,0].astype(int)

    column_names = column_names[filt]
    Xtrain_filt = np.array(Xtrain)[:, filt]
    Xtest_filt = np.array(Xtest)[:, filt]

    if verbose:
        print('Before Feat Sel: ',Xtrain.shape[1],' features')
        print('After Feat Sel: ',Xtrain_filt.shape[1],' features')
        print('------------------------')
    return filt, Xtrain_filt, Xtest_filt, column_names


def train_model(X_train, y_train, X_val, y_val, random_search_params):
    np.random.seed(15)
    random.seed(15)
    importance_array = pd.DataFrame()
    runXGB = 1
    pca = 0
    smote = 1
    verbose = False

    X_train, y_train, scaler, pca, clf, index, importance_array, column_names = classifier_train(X_train, y_train, runXGB, X_val, y_val, pca, smote, importance_array, X_train.columns, random_search_params, verbose)
    print('Feature selection using training feature importance')
    X_val = scaler.transform(X_val) #pca.transform(scaler.transform(X_val))
    imp_features, X_train, X_val, column_names = feature_selection(X_train, X_val, importance_array, column_names, verbose)
    print('Repeat training on filtered training data')
    importance_array = pd.DataFrame()
    clf, importance_array = trainXGB(X_train, y_train, X_val, y_val, importance_array, column_names, random_search_params, verbose)
    return clf, index, scaler, pca, imp_features, verbose


#Extended classification metrics

def compute_metrics(y_true, y_score, threshold=0.5):
    #All seven metrics reported in the resubmission.
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
    #Return {metric: (point, low95, high95)} from n test-set resamples.
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    N = len(y_true)
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


def save_confusion(y_true, y_score, train_payload, test_payload, out_dir):
    #Save a confusion matrix PNG using LNP_model's plot_confusion_matrix.
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    title = (f'{train_payload} model  |  {train_payload} test set')
    plot_confusion_matrix(cm, classes=['Not formed', 'Formed'],
                          title=title)
    plt.savefig(f'{out_dir}/confusion_{train_payload}_{test_payload}.svg',
                dpi=200, bbox_inches='tight')
    plt.close()

def save_predictions(y_true, y_score, train_payload, test_payload, out_dir):
    df = pd.DataFrame({
        'y_true':  np.asarray(y_true).astype(int),
        'y_score': np.asarray(y_score),
        'y_pred':  (np.asarray(y_score) > 0.5).astype(int),
    })
    df.to_csv(f'{out_dir}/predictions_{train_payload}_{test_payload}.csv', index=False)


def save_roc_pr_curves(y_true, y_score, train_payload, test_payload, out_dir):
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    precision, recall, pr_thr = precision_recall_curve(y_true, y_score)

    label = (f'{train_payload}'
            if train_payload == test_payload
            else f'train {train_payload}, test {test_payload}')


    pd.DataFrame({'fpr': fpr, 'tpr': tpr,
                  'threshold': np.append(roc_thr, np.nan)[:len(fpr)]
                  }).to_csv(f'{out_dir}/roc_curve_{train_payload}_{test_payload}.csv', index=False)
    pd.DataFrame({'precision': precision, 'recall': recall,
                  'threshold': np.append(pr_thr, np.nan)[:len(precision)]
                  }).to_csv(f'{out_dir}/pr_curve_{train_payload}_{test_payload}.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, 'b')
    axes[0].plot([0,1],[0,1],'r--')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title(f'{label} ROC')
    axes[1].plot(recall, precision, 'b')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title(f'{label} PR')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/pr_roc_{train_payload}_{test_payload}.svg', dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_feature_importance_ablation(clf, all_columns, imp_features,
                                     feature_set, out_dir, top_n=30):
    reduced_cols = np.array(all_columns)[imp_features]
    fscore = clf.get_fscore()
    rows = [{'feature': reduced_cols[int(k.replace('f', ''))], 'fscore': v}
            for k, v in fscore.items()]
    df = pd.DataFrame(rows).sort_values('fscore', ascending=False).reset_index(drop=True)
    df['fscore_normalized'] = df['fscore'] / df['fscore'].sum()
    df.to_csv(f'{out_dir}/importance_{feature_set}.csv', index=False)

    top = df.head(top_n).sort_values('fscore_normalized')
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(top['feature'], top['fscore_normalized'], color='steelblue')
    ax.set_xlabel('Relative importance')
    ax.set_title(f'{feature_set} XGBoost feature importance (top {top_n})')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/importance_{feature_set}.svg',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    return df
    
def run_ablation(feature_set):
    out_dir = f'{RESULTS_DIR}/{feature_set}'
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n=== ablation: {feature_set} ===')

    data_train = pd.read_csv(f'{DATA_DIR}/{feature_set}_train.csv')
    data_test  = pd.read_csv(f'{DATA_DIR}/{feature_set}_test.csv')

    # Drop metadata columns if present; keep label separate
    drop_cols = ['label', 'Payload', 'Lipomer', 'Cholesterol',
             'HelperLipid', 'PEGChain', 'PEG MW', 'diameter']
    y_train = np.array(data_train['label'])
    y_test  = np.array(data_test['label'])
    X_train = data_train.drop(columns=[c for c in drop_cols if c in data_train.columns])
    X_test  = data_test.drop(columns=[c for c in drop_cols if c in data_test.columns])

    with open(f'{PARAMS_DIR}/ablations/hyperparameters_{feature_set}.pkl', 'rb') as f:
        hyperparams = pickle.load(f)
    hyperparams['tree_method'] = 'exact'

    clf, index, scaler, pca, imp_features, verbose = train_model(
        X_train, y_train, X_test, y_test, hyperparams)

    X_test_scaled = np.array(scaler.transform(X_test))[:, imp_features]
    y_score, _, _ = classifier_test(X_test_scaled, y_test, clf, index,
                                    scaler, pca, verbose=False)

    metrics_ci = bootstrap_metrics(y_test, y_score)

    save_confusion(y_test, y_score, feature_set, feature_set, out_dir)
    save_predictions(y_test, y_score, feature_set, feature_set, out_dir)
    save_roc_pr_curves(y_test, y_score, feature_set, feature_set, out_dir)
    save_feature_importance_ablation(clf, X_train.columns, imp_features,
                                     feature_set, out_dir)

    return [{'feature_set': feature_set, 'metric': m,
             'point': v[0], 'low95': v[1], 'high95': v[2]}
            for m, v in metrics_ci.items()]

def plot_ablation_overlay(results_dir, feature_sets, out_dir):
    """Overlay ROC and PR curves for all ablations on a single plot."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_sets)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for fset, color in zip(feature_sets, colors):
        roc_df = pd.read_csv(f'{results_dir}/{fset}/roc_curve_{fset}_{fset}.csv')
        pr_df  = pd.read_csv(f'{results_dir}/{fset}/pr_curve_{fset}_{fset}.csv')
        preds  = pd.read_csv(f'{results_dir}/{fset}/predictions_{fset}_{fset}.csv')

        roc_auc = roc_auc_score(preds['y_true'], preds['y_score'])
        pr_auc  = average_precision_score(preds['y_true'], preds['y_score'])

        axes[0].plot(roc_df['fpr'], roc_df['tpr'], color=color,
                     label=f'{fset} (AUC={roc_auc:.3f})', linewidth=2)
        axes[1].plot(pr_df['recall'], pr_df['precision'], color=color,
                     label=f'{fset} (AP={pr_auc:.3f})', linewidth=2)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC curves across ablations')
    axes[0].legend(loc='lower right', fontsize=8)

    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('PR curves across ablations')
    axes[1].legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{out_dir}/ablation_overlay.svg', dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    rows = []
    for fset in FEATURE_SETS:
        rows.extend(run_ablation(fset))

    metrics_long = pd.DataFrame(rows)
    metrics_long.to_csv(f'{RESULTS_DIR}/metrics_long.csv', index=False)

    print('\n=== Final ablation metric panels ===')
    for m in METRIC_NAMES:
        sub = metrics_long[metrics_long['metric'] == m]
        wide = sub.set_index('feature_set')[['point', 'low95', 'high95']].round(3)
        wide.to_csv(f'{RESULTS_DIR}/metrics_{m}.csv')
        print(f'\n{m}:')
        print(wide)
    plot_ablation_overlay(RESULTS_DIR, FEATURE_SETS, RESULTS_DIR)


if __name__ == '__main__':
    main()
