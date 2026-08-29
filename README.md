# LNP Formation Predictions

This repository contains code and data associated with the manuscript, **"Machine learning predicts differences in the formation of lipid nanoparticles encapsulating mRNA or siRNA"** (Tynes et al.) currently under review. The study predicts the formation of four-component lipid nanoparticles with gradient boosted decision trees, integrated with Mordred molecular descriptors to encode chemical properties and SAPT0 ionizable lipid-cholesterol quantum-chemical interaction energies computed with AP-Net. Models are trained on nucleic acid payload specific datasets and validated on clinically inspired COVID-19 vaccine LNP chemistries.

## Dependencies

To reproduce the reported results, create a new conda environment:


```bash
conda create -n LNPdesign python=3.11 -y
conda activate LNPdesign
pip install -r requirements.txt
cd models
```

## Data

Original LNP screening data is stored in `datasets/raw_data.csv`, with one row per LNP formulation and a label indicating if a LNP formulation yields uniform particles within the desired nanoscale range.

Engineered features are stored in `datasets/processed_data.csv`, with 5 SAPT0 energy features (total, electrostatic, exchange, induction, dispersion) and approximately 1500 Mordred descriptors for each of the four components. Payload specific 80/20 splits are stored in `datasets/{mRNA,siRNA,RNA}_{train,test}.csv`.

Energy minimized lipid conformations used as input to AP-Net are computed in `models/prepare_dimers.py`. SAPT0 energies predicted with AP-Net are stored in `datasets/dimer_interactions_minenergy.pkl` for training and `datasets/dimer_interactions_pfizer_moderna.pkl` for the validation study. AP-Net source code and pretrained models are in `AP-Net-master/`.

## Preprocessing

This script `models/preprocess.py` combines SMILES strings, inferred AP-Net SAPT energies, and calculated Mordred descriptors with the molecular composition and mass ratio, then splits into payload specific datasets for mRNA, siRNA, and combined RNA train/test sets. Requires `mordred==1.2.0` or use pre-computed features in `datasets/` and skip this step to reproduce the paper's results.

```bash
python3 preprocess.py
```

## Hyperparameter tuning

This script `models/hyperparameter_search.py` runs a 100 iteration random search with 5 fold cross validation.

```bash
python3 hyperparameter_search.py
```

## Training
 
This script `models/LNP_model.py` trains a final XGBoost model with the best parameters, and scores the held-out test set. Set `model_name` to `'mRNA'`, `'siRNA'`, or `'RNA'` in the main function. The script loads tuned hyperparameters from `models/params/` to reproduce the models reported in the paper.
 
```bash
python3 LNP_model.py
```

## Inference
 
This script `models/evaluate_model.py` loads trained XGBoost models from `models/params/`, scores the held-out test set for each payload, and outputs the full metric panel with 95% bootstrap confidence intervals, confusion matrices, and ROC/PR curves to a folder `models/results/`
 
```bash
python3 evaluate_model.py
```
