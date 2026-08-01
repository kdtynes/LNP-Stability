#Build the lipomer-cholesterol dimer dataset for input to AP-Net.

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

DATA_DIR       = '../datasets'
AP_NET_DIR     = '../AP-Net-master'
DATASET_NAME   = 'LNP_minenergy'
COMPONENTS     = ['Lipomer', 'Cholesterol', 'HelperLipid', 'PEG']
ALLOWED_ATOMS  = {1, 6, 7, 8, 9, 16} # H, C, N, O, F, S
LABELS         = ['Total', 'Elst', 'Exch', 'Ind', 'Disp']
PTABLE         = Chem.GetPeriodicTable()


#Generate 3D Conformations with energy minimization
#Return MMFF94-minimized coordinates (N, 3) and atomic numbers (N,).
def smile_to_3d(smile_str, optim='MMFF94'):
    if smile_str is None or (isinstance(smile_str, float) and np.isnan(smile_str)):
        return None, None

    mol = Chem.MolFromSmiles(smile_str)
    hmol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(hmol, useRandomCoords=True)

    if optim == 'MMFF94':
        AllChem.MMFFOptimizeMolecule(hmol, mmffVariant='MMFF94', maxIters=2000)
    elif optim == 'MMFF94s':
        AllChem.MMFFOptimizeMolecule(hmol, mmffVariant='MMFF94s', maxIters=2000)
    elif optim == 'UFF':
        AllChem.UFFOptimizeMolecule(hmol, maxIters=2000)

    a = Chem.MolToMolBlock(hmol).split('\n')
    v3000 = a[4].split()[0] == 'M'
    offset = 7 if v3000 else 4

    num_atoms = hmol.GetNumAtoms()
    pos = np.zeros((num_atoms, 3))
    atom_type = np.zeros(num_atoms)
    for i in range(num_atoms):
        row = a[offset + i].split()
        pos[i] = row[4:7] if v3000 else row[:3]
        atom_type[i] = PTABLE.GetAtomicNumber(row[3])
    return pos, atom_type

#Append MMFF94-minimized 'R' and 'Z' columns for every unique SMILES.
def compute_conformers(smiles_ref):
    tqdm.pandas()
    smiles_ref['Positions'] = smiles_ref['SMILES'].progress_apply(smile_to_3d)
    smiles_ref[['R', 'Z']] = pd.DataFrame(smiles_ref['Positions'].tolist(),
                                          index=smiles_ref.index)
    return smiles_ref


#Formulation SMILES
#Attach a '{Component} SMILES' column for each of the four components.
def attach_component_smiles(raw, smiles_ref):
    raw = raw.rename(columns={'Helper Lipid': 'HelperLipid'}).copy()
    raw['PEG'] = raw['PEGChain'].str.strip() + 'PEG' + raw['PEG MW'].astype(str)

    lookup = smiles_ref.set_index('Name')['SMILES']
    for col in COMPONENTS:
        raw[col + ' SMILES'] = raw[col].map(lookup)

    smiles_cols = [c + ' SMILES' for c in COMPONENTS]
    return raw.dropna(subset=smiles_cols).reset_index(drop=True)


#Construct dimers
#Unique (A, B) SMILES pairs for the given pair of component indices.
def build_dimers(smiles, pair_idx=(0, 1)):
    smiles_cols = [c + ' SMILES' for c in COMPONENTS]
    comb_arr = smiles[smiles_cols].iloc[:, list(pair_idx)]
    dimers = np.unique(comb_arr.to_numpy().astype(str), axis=0)
    return pd.DataFrame(dimers, columns=['A', 'B'])

#Join R (positions) and Z (atomic numbers) onto both monomers of each dimer.
def join_coordinates(dimers, smiles_ref):
    ref = smiles_ref[['Name', 'SMILES', 'R', 'Z']].copy()
    ref['Z'] = ref['Z'].apply(lambda x: list(map(int, x)))
    ref = ref.drop_duplicates(subset='SMILES').set_index('SMILES')

    for i, col in enumerate(['A', 'B']):
        monomer = 'A' if i == 0 else 'B'
        dimers = dimers.join(ref, how='left', on=col)
        dimers = dimers.rename(columns={'R': 'R' + monomer,
                                        'Z': 'Z' + monomer,
                                        'Name': 'Name_' + monomer})

    print(np.argwhere(np.array(dimers['RA'].isnull()) == True))
    dimers = dimers[dimers['RA'].isnull() == False]           # if R is null, Z is null
    dimers = dimers[dimers['RB'].isnull() == False]
    return dimers.reset_index(drop=True)

#Shift monomer B by monomer A's bounding box so the two do not overlap.
def separate_monomers(data):
    for i in tqdm(range(len(data))):
        RA = np.array(data['RA'][i])
        RB = np.array(data['RB'][i])

        box_x = np.max(RA[:, 0]) - np.min(RA[:, 0])
        box_y = np.max(RA[:, 1]) - np.min(RA[:, 1])
        box_z = np.max(RA[:, 2]) - np.min(RA[:, 2])

        RB[:, 0] += box_x
        RB[:, 1] += box_y
        RB[:, 2] += box_z
        data['RB'][i] = RB
    return data

#Substitute any atom outside ALLOWED_ATOMS with carbon; log each swap.
def _swap_invalid_atoms(Z, dimer_idx, mol_name, monomer, invalid_atoms):
    print('----before----')
    print(np.unique(Z))
    new_Z = Z.copy()
    for idx, atm in enumerate(Z):
        if atm not in ALLOWED_ATOMS:
            new_Z[idx] = 6
            invalid_atoms.add(atm)
            old_atom = Chem.Atom(atm).GetSymbol()
            new_atom = Chem.Atom(6).GetSymbol()
            print(f'idx {dimer_idx}: Replacing {old_atom} with {new_atom} '
                  f'in molecule {mol_name}')
    print('----after----')
    print(np.unique(new_Z))
    return new_Z

#Replace atoms outside AP-Net's training composition {H,C,N,O,F,S} with carbon.
def replace_atoms(data):
    invalid_atoms = set()
    print(data.shape)
    for i in tqdm(range(len(data))):
        for monomer in ('A', 'B'):
            Z = data[f'Z{monomer}'][i]
            if bool(set(Z) - ALLOWED_ATOMS) is True:
                data[f'Z{monomer}'][i] = _swap_invalid_atoms(
                    Z, i, data[f'Name_{monomer}'][i], monomer, invalid_atoms)
    print(data.shape)
    print(invalid_atoms)
    return data

    #Add zero placeholder labels, coerce array columns, drop name columns.
def finalize_dimers(data):
    for label in LABELS:
        data[label] = 0
    for col in ['RA', 'RB', 'ZA', 'ZB']:
        data[col] = data[col].apply(np.array)
    return data.drop(columns=['Name_A', 'Name_B'])

def main():
    # Load raw screening data and SMILES reference table
    raw        = pd.read_csv(f'{DATA_DIR}/raw_data.csv')
    smiles_ref = pd.read_csv(f'{DATA_DIR}/Smiles_complete.csv').drop_duplicates('Name')

    # Attach a SMILES column for each of the four components per formulation
    smiles = attach_component_smiles(raw, smiles_ref)
    smiles.to_csv(f'{DATA_DIR}/data_smiles.csv', index=False)
    print(smiles)

    # Compute 3D coordinates for every unique component SMILES
    smiles_ref = compute_conformers(smiles_ref)

    # Build unique lipomer-cholesterol dimers and attach coordinates
    dimers = build_dimers(smiles, pair_idx=(0, 1))
    print(dimers)
    data = join_coordinates(dimers, smiles_ref)

    # Separate monomers in space, then sanitize atoms outside AP-Net's domain
    data = separate_monomers(data)
    data = replace_atoms(data)
    data = finalize_dimers(data)

    # Export AP-Net input pickle
    out_dir = f'{AP_NET_DIR}/datasets/{DATASET_NAME}'
    os.makedirs(out_dir, exist_ok=True)
    data.to_pickle(f'{out_dir}/dimers.pkl')

if __name__ == '__main__':
    main()
