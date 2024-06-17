import torch
import esm
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# Load ESM-2 model
import json
from collections import OrderedDict


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

dataset_name="biosnap"

# dataset = json.load(open('./'+dataset_name+'_proteins.txt'),object_pairs_hook = OrderedDict)
dataset = np.loadtxt('./'+dataset_name+'/protein.txt', dtype=str)

max_len = 1024
data=[]

for i in range(len(dataset)):
    seq=dataset[i]
    if len(seq) > max_len-2:
        seq = seq[:max_len-2]
    data.append((i, seq))

# for ii in dataset.keys():
#     seq=dataset[ii]
#     if len(seq) > max_len-2:
#         seq = seq[:max_len-2]
#     data.append((1, seq))


batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#
print(batch_tokens.size())

# batch_tokens = batch_tokens.cuda()
# batch_lens = batch_lens.cuda()
# model=model.cuda()
#
train_data = TensorDataset(batch_tokens, batch_lens)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)


# Extract per-residue representations (on CPU)
with torch.no_grad():
    token_representations = torch.Tensor()#.cuda()
    for i, item in enumerate(train_loader):
        print(i)
        tok, bat = item
        results = model(tok, repr_layers=[33], return_contacts=True)
        token_rep = results["representations"][33]
        print(token_rep.size())
        # print(bat.size())
        ss = token_rep[0, 1: bat[0] - 1].mean(0).unsqueeze(0)
        # print(ss.size())
        token_representations=torch.cat((token_representations, ss), 0)
print(token_representations.size())


sequence_representations = token_representations.cpu().numpy()
np.savetxt("./esm_"+dataset_name+".txt", sequence_representations, delimiter=',', fmt="%.8f")



