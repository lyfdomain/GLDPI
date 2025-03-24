import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem



def morgan_smiles(line, dim_num):
    mol = Chem.MolFromSmiles(line)
    feat = AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim_num)

    return feat


def sim_recon(S, t):
    sorted_drug = (S).argsort(axis=1).argsort(axis=1)
    np.fill_diagonal(sorted_drug, 0)
    sorted_drug = (len(S) - 1) * np.ones((len(S), len(S))) - sorted_drug
    sorted_drug[sorted_drug == 0] = 1
    sorted_drug = 1 / ((sorted_drug) ** (1 / t))  # *(sorted_drug+1))
    np.fill_diagonal(sorted_drug, 1)
    SS = (sorted_drug + sorted_drug.T) / 2
    return SS
