#Evaluate the mRNA, siRNA, and RNA XGBoost models on the COVID-19 vaccine
#validation set (Pfizer and Moderna formulations)

import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd

from LNP_model import train_model, classifier_test

warnings.filterwarnings('ignore')

DATA_DIR       = '../datasets'
VALIDATION_DIR = '../datasets/validation_study'
PARAMS_DIR     = './params'
RESULTS_DIR    = './validation_results'

PAYLOADS       = ['mRNA', 'siRNA', 'RNA']
VACCINES       = ['pfizer', 'moderna']
SEED           = 15

DROP_COLS = ['label', 'Lipomer', 'Cholesterol', 'HelperLipid',
             'PEGChain', 'PEG MW', 'diameter', 'Payload']

#Load the payload-specific train/test split used to fit the model.
def load_training(payload):
    train = pd.read_csv(f'{DATA_DIR}/{payload}_train.csv')
    test  = pd.read_csv(f'{DATA_DIR}/{payload}_test.csv')
    for df in (train, test):
        if 'Payload' in df.columns:
            df.drop(columns=['Payload'], inplace=True)
    y_train = np.array(train['label'])
    y_test  = np.array(test['label'])
    X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    X_test  = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    return X_train, y_train, X_test, y_test

#Load a validation vaccine CSV and align columns to the training set.
def load_validation(vaccine, train_columns):
    path = f'{VALIDATION_DIR}/data_{vaccine}.csv'
    data = pd.read_csv(path)

    X = data[train_columns].copy()
    X = clean_mordred_errors(X)

    # Optional label column for reference; not used in scoring
    y = np.array(data['label']) if 'label' in data.columns else None
    return X, y, data

#Clean errors by forcing object columns to numeric, and replace inf/nan with 0
def clean_mordred_errors(data):
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.fillna(0)

#Apply the trained pipeline to a validation set, return prediction scores.
def score_validation(X_val, y_val, clf, scaler, pca, imp_features):
    X_scaled = scaler.transform(X_val)
    X_filt   = np.array(X_scaled)[:, imp_features]

    pred, _, _ = classifier_test(X_filt, y_val, clf, 1, scaler, pca, verbose=False)
    return pred


#Train a payload specific models, then validate on vaccines."""
def run_payload_validation(payload):
    print(f'\n=== training {payload} model ===')
    X_train, y_train, X_test, y_test = load_training(payload)

    with open(f'{PARAMS_DIR}/hyperparameters{payload}.pkl', 'rb') as f:
        hyperparams = pickle.load(f)
    hyperparams['tree_method'] = 'exact'

    clf, index, scaler, pca, imp_features, verbose = train_model(
        X_train, y_train, X_test, y_test, hyperparams)
    print(f'  trained; selected {len(imp_features)} of {X_train.shape[1]} features')

    train_columns = X_train.columns.tolist()

    for vaccine in VACCINES:
        print(f'\n  scoring {vaccine} formulations')
        X_val, y_val, original = load_validation(vaccine, train_columns)
        y_score = score_validation(X_val, y_val, clf, scaler, pca, imp_features)

        original.insert(0, f'prediction_{payload}',
                        (y_score > 0.5).astype(int))
        original.insert(0, f'score_{payload}', y_score)

        out_path = f'{RESULTS_DIR}/{vaccine}_predictions_{payload}.csv'
        original.to_csv(out_path, index=False)
        print(f'    saved: {out_path}')
        print(f'    score range: [{y_score.min():.3f}, {y_score.max():.3f}], '
              f'mean {y_score.mean():.3f}')

#Outputs a CSV with predictions for each vaccine.
def combine_predictions():
    for vaccine in VACCINES:
        dfs = []
        for payload in PAYLOADS:
            path = f'{RESULTS_DIR}/{vaccine}_predictions_{payload}.csv'
            df = pd.read_csv(path)
            dfs.append(df[[f'score_{payload}', f'prediction_{payload}']])

        base = pd.read_csv(f'{RESULTS_DIR}/{vaccine}_predictions_{PAYLOADS[0]}.csv')
        base = base.drop(columns=[f'score_{PAYLOADS[0]}',
                                   f'prediction_{PAYLOADS[0]}'])

        combined = pd.concat([base] + dfs, axis=1)
        out_path = f'{RESULTS_DIR}/{vaccine}_predictions_all_models.csv'
        combined.to_csv(out_path, index=False)
        print(f'  wrote combined: {out_path}')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    for payload in PAYLOADS:
        run_payload_validation(payload)

    print('\n=== combining predictions across payload models ===')
    combine_predictions()


if __name__ == '__main__':
    main()