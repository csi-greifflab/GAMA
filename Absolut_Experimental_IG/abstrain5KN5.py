# COMPUTE IG
import os
import sys
from statistics import mean

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import pandas as pd

sys.path.append('/fp/projects01/ec195/storage/rofrank/absolutTrain')
from igLSTM import LSTMModel

DEVICE = 'cuda:0'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#mHER_H3_AgPos_unique.csv
df = pd.read_csv('/fp/projects01/ec195/storage/rofrank/absolutTrain/5KN5_C_A_MascotteSlices_feature.tsv', sep='\t', header=1)
#df = pd.read_csv('5KN5_C_A_MascotteSlices_feature.tsv', sep='\t', header=1)
#epmask = '1011011000121030001120000000000000000000'
epmask = '1011011000121030001120000000000000000000'

df = df.query('interMaskAGEpitope == @epmask').reset_index(drop=True)

alphabet = 'ACDEFGHIKLMNPQRSTVWY'
encoding = {letter:i for i, letter in enumerate(alphabet, start=1)}
reverse_encoding = {i:letter for i, letter in enumerate(alphabet)}
train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]).to(DEVICE) for i in df.Slide.tolist()]

model = LSTMModel(1024, 22, DEVICE).to(DEVICE)
torch.save(model.state_dict(), f'/fp/projects01/ec195/storage/rofrank/absolutTrain/train5KN5/5KN5_Init_model.pt')
optimizer = optim.Adam(model.parameters(),lr=0.00009)
loss_function = nn.NLLLoss().to(DEVICE)
train_data = DataLoader(train_list, shuffle=True, batch_size=None)

train_loss = list()

for epoch in range(1_000_000):
    loss_o = list()
    for sentence in train_data:
        model.zero_grad(set_to_none=True)
        tag_scores = model(sentence[0:-1])
        loss = loss_function(tag_scores.view(-1, 22), sentence[1:])
        loss.backward()
        optimizer.step()
        loss_o.append(loss.detach().tolist())
    #train_loss.append(mean(loss_o))
    #train_loss[train_loss.shape[1]] = loss_o
    #print(f'{epoch=}, {mean(loss_o)=}')
    #print(f'{epoch=}, {mean(loss_o)=}')
    torch.save(model.state_dict(), f'/fp/projects01/ec195/storage/rofrank/absolutTrain/train5KN5/lstm_epoch_{epoch}_{mean(loss_o)}.pt')
#pd.DataFrame(train_loss).to_csv('erroruig5CZV0_000001.tsv', sep='\t', index=False)