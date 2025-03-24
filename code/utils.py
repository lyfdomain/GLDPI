import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# Calculate the Morgan fingerprints of drugs
def morgan_smiles(line, dim_num):
    mol = Chem.MolFromSmiles(line)
    feat = AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim_num)
    return feat

# Reconstruct the similarities between proteins
def sim_recon(S, t):
    sorted_pro = (S).argsort(axis=1).argsort(axis=1)
    np.fill_diagonal(sorted_pro, 0)
    sorted_pro = (len(S) - 1) * np.ones((len(S), len(S))) - sorted_pro
    sorted_pro[sorted_pro == 0] = 1
    sorted_pro = 1 / ((sorted_pro) ** (1 / t))  # *(sorted_pro+1))
    np.fill_diagonal(sorted_pro, 1)
    SS = (sorted_pro + sorted_pro.T) / 2
    return SS
