import random
import torch
import numpy as np
from torch import nn
import csv
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from torch.nn import functional
from utils import *

class Autoencoder(nn.Module):
    def __init__(self, input_size, input_size2, hidden_size):
        super(Autoencoder, self).__init__()

    def forward(self, x, y):
        encoded = self.encoder(x)
        decoded = self.decoder(y)
        return encoded, decoded

jihe = "biosnap"
morgan_dim = 1024
model = torch.load("predti.pth")
model.eval()

dti = np.loadtxt("./dataset/"+ jihe +"/dti.txt").astype(dtype="int64")
drug = np.loadtxt("./dataset/"+ jihe +"/drug.txt", dtype=str, comments=None)
protein = np.loadtxt("./dataset/"+ jihe +"/protein.txt", dtype=str)

drug_data = []
with open("./dataset/" + jihe + "/test.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        # print(', '.join(row))
        drug_data.append(row)
drug_data = np.array(drug_data)

ind = []
for i in range(len(drug_data)):
    if drug_data[i, 0] in drug and drug_data[i, 1] in protein:
        ind.append([list(drug).index(drug_data[i, 0]), list(protein).index(drug_data[i, 1])])

tt = []
for i in range(len(drug)):
    xd = morgan_smiles(drug[i], morgan_dim)
    tt.append(xd)
tt = np.array(tt)

pp = np.loadtxt("./dataset/"+ jihe +"/esm2_"+ jihe +".txt", delimiter=",")

x=torch.from_numpy(tt).float()
y=torch.from_numpy(pp).float()

output1, output2 = model(x,y)
output1=output1.detach().numpy()
output2=output2.detach().numpy()

pre=cosine_similarity(output1,output2)

pr=[]
tr=[]
for i in range(len(ind)):
    pr.append(pre[ind[i][0],ind[i][1]])
    tr.append(dti[ind[i][0],ind[i][1]])

y_label = tr
y_pred = pr

preci, recal, thres = precision_recall_curve(y_label, y_pred)
best_thre_id = np.argmax(2*preci * recal/((preci + recal)+1e-8))

best_thre = thres[best_thre_id]
y_pred_s = [1 if i else 0 for i in (y_pred >= best_thre)]
accuracy = accuracy_score(y_label, y_pred_s)
print("Acc", accuracy)
f1score = f1_score(y_label, y_pred_s)
print("F1", f1score)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(tr, pr)
area = sklearn.metrics.auc(fpr, tpr)
print("the Area Under the ROCurve is:", area)
aps = average_precision_score(tr, pr)
print("the AP score is:", aps)

