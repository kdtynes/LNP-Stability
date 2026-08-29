#Preprocess raw LNP screening data into per-payload train/test splits

import pickle
import random
import warnings

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

warnings.filterwarnings('ignore')


def data_preprocessing(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Renaming columns
    data = data.rename(columns={'Included in screen?': 'classifier'})
    data = data.rename(columns={'Helper Lipid': 'HelperLipid'})
    data = data.rename(columns={'Diameter (nm)': 'diameter'})

    # Drop metadata columns
    data = data.drop(columns=['LNP #', 'Date', 'Initials', 'Project',
                              'Screen Successful?', 'Confirmed?',
                              'Confirmation Success', 'Cell Types'],
                     errors='ignore')

    # Strip whitespace
    data['PEG']   = data['PEG'].str.strip()
    data['Payload']    = data['Payload'].str.strip()
    data['classifier'] = data['classifier'].str.strip()

    # Convert to binary classification {0, 1}
    data['classifier'] = data['classifier'].map({'Y': 1, 'N': 0})
    data = data[data.classifier.isnull() == False].reset_index(drop=True)

    # In the HelperLipid column '-' are equivalent to 'None' (Helper Lipid Mole % = 0)
    data.loc[data.HelperLipid == '-',    'HelperLipid'] = 'None'
    # Replace null helper lipids with the lipomer
    data.loc[data.HelperLipid == 'None', 'HelperLipid'] = data.Lipomer
    data = data.reset_index(drop=True)

    # Ordinal-encode payload (mRNA=0, siRNA=1, bcDNA=2)
    data.loc[data['Payload'].str.contains('mRNA'), 'Payload'] = "0"  # mRNA
    data.loc[data['Payload'].str.contains('aVHH'), 'Payload'] = "0"  # aVHH treated as mRNA
    data.loc[data['Payload'].str.contains('si'),   'Payload'] = "1"  # siRNA
    data.loc[data['Payload'].str.contains('BC'),   'Payload'] = "2"  # DNA barcode
    data['Payload'] = pd.to_numeric(data['Payload'], errors='coerce')

    # Separate data and labels
    y = np.array(data['classifier'])
    data = data.drop(columns=['classifier'])
    N, D = data.shape

    # Report class balance
    vc = np.unique(y, return_counts=True)
    for i in range(len(vc[0])):
        print(f"Number of {vc[0][i]}'s: {vc[1][i]} ({100*vc[1][i]/N:.1f}%)")
    print(f"Number of data points: {N}  ::  Number of features: {D}\n")
    return data, y


def add_smiles(data, smiles):
    cols_to_smiles = ['Lipomer', 'Cholesterol', 'HelperLipid', 'PEG']
    for col in cols_to_smiles:
        data = data.join(smiles.set_index('Name'), on=col)
        new_name = col + ' SMILES'
        data = data.rename(columns={'SMILES': new_name})

    # Keep only rows with non-null SMILES for all four components
    smiles_cols = [c + ' SMILES' for c in cols_to_smiles]
    data = data.dropna(subset=smiles_cols).reset_index(drop=True)
    print(f"Rows with complete SMILES: {len(data)}")
    return data


def add_interactions(data, interactions):
    """Join SAPT0 energies (Total/Elst/Exch/Ind/Disp) on the lipomer-cholesterol pair."""
    joined_data = pd.merge(data, interactions, how='left',
                           left_on=['Lipomer SMILES', 'Cholesterol SMILES'],
                           right_on=['A', 'B'])
    joined_data = joined_data.drop(columns=['A', 'B', 'RA', 'ZA', 'RB', 'ZB'],
                                   errors='ignore')
    return joined_data


def add_mordred(data, mordred_df):
    """Join Mordred descriptors for each component, suffixed with the component name."""
    cols_to_smiles = ['Lipomer', 'Cholesterol', 'HelperLipid', 'PEG']
    for col in cols_to_smiles:
        feat = col + ' SMILES'
        temp = mordred_df.copy()
        temp = temp.drop(columns=['Name'])
        temp = temp.drop_duplicates(subset='SMILES')

        # Suffix each descriptor column with the component name
        drop = temp['SMILES']
        temp = temp.drop(columns=['SMILES'])
        temp = temp.add_suffix('_' + col)
        temp.insert(0, 'SMILES', drop)
        data = data.join(temp.set_index('SMILES'), how='left', on=feat)
    data = data.drop(columns=['Lipomer SMILES', 'Cholesterol SMILES',
                              'HelperLipid SMILES', 'PEG SMILES'])
    return data


def remove_mordred_errors(data):
    #Coerce object-typed Mordred columns to numeric; drop columns that become all-zero.
    # Select object columns, excluding the first 12 non-mordred columns
    obj_cols = data.iloc[:, 12:].select_dtypes(include='object').columns
    
    cols_to_remove = []
    for col in obj_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        if (data[col] == 0).all():
            cols_to_remove.append(col)
    
    print(f"num cols removed: {len(cols_to_remove)}")
    data = data.drop(columns=cols_to_remove)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.fillna(0)


def train_test_split(data):
    """80/20 split with a fixed seed so the same rows go to train/test every run."""
    N = data.shape[0]
    train_N = int(np.ceil(0.8 * N))

    random.seed(33333)
    np.random.seed(33333)
    shuffle_data = np.arange(N)
    random.shuffle(shuffle_data)

    training_rows = shuffle_data[:train_N]
    testing_rows  = shuffle_data[train_N:]

    data_train = data.iloc[training_rows].reset_index(drop=True)
    data_test  = data.iloc[testing_rows].reset_index(drop=True)
    return data_train, data_test


def check_imbalance(data, name):
    vc = np.unique(data['label'], return_counts=True)
    N, D = data.shape
    print(f"--- {name} ---")
    for i in range(len(vc[0])):
        print(f"Number of {vc[0][i]}'s: {vc[1][i]} ({100*vc[1][i]/N:.1f}%)")
    print(f"Number of data points: {N}  ::  Number of features: {D}\n")

def main():
    DATA_DIR = '../datasets'   # script lives in Models/, data in Datasets/

    # Load raw data
    data = pd.read_csv(f'{DATA_DIR}/raw_data.csv')
    data, y = data_preprocessing(data)

    # Append SMILES strings for each component
    smiles = pd.read_csv(f'{DATA_DIR}/Smiles_complete.csv')[['Name', 'SMILES']]
    smiles = smiles.drop_duplicates('Name')
    data = add_smiles(data, smiles)

    # Append SAPT0 interaction energies (min energy conformations -> AP-Net)
    with open(f'{DATA_DIR}/dimer_interactions_minenergy.pkl', 'rb') as f:
        interactions = pickle.load(f)
    data = add_interactions(data, interactions)

    # Append Mordred descriptors (~1500 per component)
    #mordred_df = pd.read_csv(f'{DATA_DIR}/mordred_df.csv')
    calc = Calculator(descriptors, ignore_3D=True)
    mols = smiles['SMILES'].apply(Chem.MolFromSmiles)
    chem_df = calc.pandas(mols)
    mordred_df = pd.concat([smiles.reset_index(drop=True),chem_df.reset_index(drop=True)],axis=1)
    data = add_mordred(data, mordred_df)

    # Clean Mordred output
    data = remove_mordred_errors(data)

    # Reattach labels and write the processed feature table
    data.insert(0, 'label', y)
    data.to_csv(f'{DATA_DIR}/processed_data.csv', index=False)

    # Split by payload (mRNA, siRNA) and combine for the RNA model
    mRNA_data  = data.loc[data['Payload'] == 0]
    siRNA_data = data.loc[data['Payload'] == 1]

    check_imbalance(mRNA_data, 'mRNA')
    mRNA_train, mRNA_test = train_test_split(mRNA_data)
    mRNA_train.to_csv(f'{DATA_DIR}/mRNA_train.csv', index=False)
    mRNA_test.to_csv(f'{DATA_DIR}/mRNA_test.csv',  index=False)

    check_imbalance(siRNA_data, 'siRNA')
    siRNA_train, siRNA_test = train_test_split(siRNA_data)
    siRNA_train.to_csv(f'{DATA_DIR}/siRNA_train.csv', index=False)
    siRNA_test.to_csv(f'{DATA_DIR}/siRNA_test.csv',  index=False)

    # Combined RNA model (mRNA + siRNA, shuffled together)
    RNA_train = pd.concat([mRNA_train, siRNA_train], ignore_index=True)
    RNA_test  = pd.concat([mRNA_test,  siRNA_test],  ignore_index=True)
    RNA_train = RNA_train.sample(frac=1, random_state=33333).reset_index(drop=True)
    RNA_test  = RNA_test.sample(frac=1, random_state=33333).reset_index(drop=True)
    check_imbalance(pd.concat([RNA_train, RNA_test]), 'RNA (combined)')
    RNA_train.to_csv(f'{DATA_DIR}/RNA_train.csv', index=False)
    RNA_test.to_csv(f'{DATA_DIR}/RNA_test.csv',  index=False)


if __name__ == '__main__':
    main()
