# Accurate prediction of drug-protein interactions by maintaining the original topological relationships among embeddings (GLDPI)

This is a PyTorch implementation of GLDPI, and the code includes the following files:

* data_process.py

*  model.py

*  untils.py

*  train.py


environment:
python3.9 pytorch1.11.0+cu11.3

1.Extract esm_biosnap.zip to obtain the protein esm2 coding file  esm_biosnap.txt

2.Ran data_process.py to obtain drug similarity matrix DS.txt, protein similarity matrix PS.txt, drug SMILES list drug.txt, protein sequences list protein.txt and drug-protein interaction matrix dti.txt.

3.Run train.py to get the result.

Contacts
If you have any questions, please email Li Yanfei (lyfinf@163.com)
