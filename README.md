# GLDPI
Accurate prediction of drug-protein interactions by maintaining the original topological relationships among embeddings 

# Main Files
This is a PyTorch implementation of GLDPI, and the code includes the following files:

* data_preprocess.py
    - Generate the corresponding drug SMILES list and protein amino acid sequence list based on datasets (biosnap et al.), and compute the drug similarity matrix based on the drug SMILES, as well as the protein similarity matrix based on the protein amino acid sequences.
*  model.py
    - Set prediction framework and GBA_loss
*  utils.py
    - morgan_smiles()--Calculate the Morgan fingerprints of drugs
    - sim_recon()--Reconstruct the similarities between proteins
*  train.py
    - Set parameters and train the model
*  prediction.py
    - Inference and evaluat predictive performance
# Main Requirements

* python==3.9 
* pytorch==1.11.0+cu11.3
* numpy==1.26.2
* sklearn== 1.3.2


# Dataset

* The BindingDB dataset can be acquired at https://github.com/peizhenbai/DrugBAN/main/datasets.
* The Biosnap dataset can be obtained at https://github.com/samsledje/ConPLex_duev/tree/main/dataset/BIOSNAP.
* The Davis dataset can be found at https://github.com/hkmztrk/DeepDTA/tree/master/data/davis.

# Train

1. Run data_preprocess.py to generate the required intermediate files

2. Run train.py to train GLDPI model

# Prediction 

Run prediction.py to test and evaluate the prediction performance of GLDPI model
