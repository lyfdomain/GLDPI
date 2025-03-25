import copy

import torch
import csv
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from model import *
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(10)


############  Read the training set and validation set data   ###############

jihe = "dataset/biosnap"
drug_data=[]

with open("./"+jihe+"/train.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        # 处理每一行数据
        drug_data.append(row)
drug_data= np.array(drug_data)

val_data = []
with open("./" + jihe + "/val.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        val_data.append(row)
val_data = np.array(val_data)


dti = np.loadtxt("./"+jihe+"/dti.txt").astype(dtype="int64")
SR = np.loadtxt('./'+jihe+'/DS.txt')
SP = np.loadtxt('./'+jihe+'/PS.txt')
SP = sim_recon(SP, 3)

#####################  Set relevant parameters   ###########################

leraning_rate=0.00001
hidden_size = 512
num_epochs = 2000
morgan_dim = 1024


dru = np.loadtxt('./'+jihe+'/drug.txt', dtype=str, comments=None)
pro = np.loadtxt('./'+jihe+'/protein.txt', dtype=str)

drug=list(dru)
protein=list(pro)

pp=np.loadtxt('./'+jihe+'/esm2_'+jihe+'.txt', delimiter=",")

############  Calculat the Morgan Fingerprints of Drugs   #################

tt = []
for i in range(len(dru)):
    xd = morgan_smiles(dru[i], morgan_dim)
    tt.append(xd)
tt = np.array(tt)


ind_val = []
for i in range(len(val_data)):
    if val_data[i, 0] in drug and val_data[i, 1] in protein:
        ind_val.append([drug.index(val_data[i, 0]), protein.index(val_data[i, 1])])

################  Collect ground truth labels  #############################

tr = []
for i in range(len(ind_val)):
    tr.append(dti[ind_val[i][0], ind_val[i][1]])

auc_max=0

##############  Collect prediction values  ###################################
def test(e, f):
    output1 = e.detach().cpu().numpy()
    output2 = f.detach().cpu().numpy()
    pre = cosine_similarity(output1, output2)
    pr = []
    for i in range(len(ind_val)):
        pr.append(pre[ind_val[i][0], ind_val[i][1]])
    return pr

cof = np.zeros(dti.shape)

############  Collect training data index  ###################################

ind = []
for i in range(len(drug_data)):
    if drug_data[i,0] in drug and drug_data[i,1] in protein:
        ind.append([drug.index(drug_data[i,0]), protein.index(drug_data[i,1])])


############  Mask non-training data  #########################################
for i in range(len(ind)):
    cof[ind[i][0], ind[i][1]] = 1


dti = torch.from_numpy(dti).float().cuda()
cof = torch.from_numpy(cof).float().cuda()
x = torch.from_numpy(tt).float().cuda()
y = torch.from_numpy(pp).float().cuda()
sr = torch.from_numpy(SR).float().cuda()
sp = torch.from_numpy(SP).float().cuda()

# 创建自动编码器模型
input_size = len(tt[0])
input_size2 = len(pp[0])
autoencoder = Autoencoder(input_size, input_size2, hidden_size)

############  Define the loss function and optimizer ##########################

autoencoder = autoencoder.cuda()
model_max = copy.deepcopy(autoencoder)
criterion = GBALoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=leraning_rate)

##########################  Train model #######################################
for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()
    e, f = autoencoder(x, y)
    loss1 = criterion(e, sr, f, sp, dti, cof)
    loss = 1 * loss1
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.cpu().item()))

    autoencoder.eval()
    pr = test(e, f)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(tr, pr)
    area = sklearn.metrics.auc(fpr, tpr)
    aps = average_precision_score(tr, pr)
    if area > auc_max:
        auc_max = area
        model_max = copy.deepcopy(autoencoder)
    print(area, aps, "AUC_max", auc_max)
    
##########################  Save model #######################################
torch.save(model_max.cpu(), 'predti.pth')



