"""
The script trains an LSTM model on CDRH3 sequences of HER-2 binders.
"""

import os
import sys
from statistics import mean

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

sys.path.append('..')
from igL_lstm import LSTMModel


DEVICE = 'cuda:1'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#mHER_H3_AgPos_unique.csv
df = pd.read_csv("mHER_H3_AgPos_unique.csv")
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
reverse_encoding = dict(enumerate(ALPHABET, start=1))
encoding = dict(zip(reverse_encoding.values(), reverse_encoding.keys()))
train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]).to(DEVICE) for i in df.AASeq.tolist()]

model = LSTMModel(1024, DEVICE).to(DEVICE)
torch.save(model.state_dict(), 'modelmasonSEToken2_long/Mason_Init_model.pt')
optimizer = optim.Adam(model.parameters())
loss_function = nn.NLLLoss().to(DEVICE)
train_data = DataLoader(train_list, shuffle=True, batch_size=None)

train_loss = pd.DataFrame()

for epoch in range(1_000_000):
    loss_o = []
    for sentence in train_data:
        model.zero_grad(set_to_none=True)
        tag_scores = model(sentence[0:-1])
        loss = loss_function(tag_scores.view(-1, 22), sentence[1:])
        loss.backward()
        optimizer.step()
        loss_o.append(loss.detach().tolist())
    train_loss[train_loss.shape[1]] = loss_o
    print(f'{epoch=}, {mean(loss_o)=}')
    torch.save(model.state_dict(), f'modelmasonSEToken2_long/lstm_epoch_{epoch}_{mean(loss_o)}.pt')
