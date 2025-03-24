# Accurate prediction of drug-protein interactions by maintaining the original topological relationships among embeddings (GLDPI)

This is a PyTorch implementation of GLDPI, and the code includes the following files:

* data_process.py

*  model.py

*  untils.py

*  train.py


# Main Requirements

* python==3.9 
* pytorch==1.11.0+cu11.3
* numpy==1.26.2
* sklearn== 1.3.2


# Dataset

* The BindingDB dataset can be acquired at https://github.com/peizhenbai/DrugBAN/main/datasets.
* The Biosnap dataset can be obtained at https://github.com/samsledje/ConPLex_dev/tree/main/dataset/BIOSNAP.
* The Davis dataset can be found at https://github.com/hkmztrk/DeepDTA/tree/master/data/davis.

# Ran the code

1.Extract esm_biosnap.zip to obtain the protein esm2 coding file  esm_biosnap.txt

2.Ran data_process.py to obtain drug similarity matrix DS.txt, protein similarity matrix PS.txt, drug SMILES list drug.txt, protein sequences list protein.txt and drug-protein interaction matrix dti.txt.

3.Run train.py to get the result.

Contacts
If you have any questions, please email Li Yanfei (lyfinf@163.com)
