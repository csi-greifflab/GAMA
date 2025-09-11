"""
The script loads an Absolut! file with antigen-specific CDRH3 sequences, filters them for
a specific epitope interaction mask, and trains an LSTM model on the sequences.
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
from ig_lstm import LSTMModel

DEVICE = 'cuda:0'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#mHER_H3_AgPos_unique.csv
df = pd.read_csv('/fp/projects01/ec195/storage/rofrank/absolutTrain/5KN5_C_A_MascotteSlices_feature.tsv', sep='\t', header=1)
#df = pd.read_csv('5KN5_C_A_MascotteSlices_feature.tsv', sep='\t', header=1)
#EPMASK = '1011011000121030001120000000000000000000'
EPMASK = '1011011000121030001120000000000000000000'

df = df.query('interMaskAGEpitope == @EPMASK').reset_index(drop=True)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
reverse_encoding = dict(enumerate(ALPHABET, start=1))
encoding = dict(zip(reverse_encoding.values(), reverse_encoding.keys()))
train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]).to(DEVICE) for i in df.Slide.tolist()]

model = LSTMModel(1024, DEVICE).to(DEVICE)
torch.save(model.state_dict(), '/fp/projects01/ec195/storage/rofrank/absolutTrain/train5KN5/5KN5_Init_model.pt')
optimizer = optim.Adam(model.parameters(),lr=0.00009)
loss_function = nn.NLLLoss().to(DEVICE)
train_data = DataLoader(train_list, shuffle=True, batch_size=None)

train_loss = []

for epoch in range(1_000_000):
    loss_o = []
    for sentence in train_data:
        model.zero_grad(set_to_none=True)
        tag_scores = model(sentence[0:-1])
        loss = loss_function(tag_scores.view(-1, 22), sentence[1:])
        loss.backward()
        optimizer.step()
        loss_o.append(loss.detach().tolist())
    torch.save(model.state_dict(), f'/fp/projects01/ec195/storage/rofrank/absolutTrain/train5KN5/lstm_epoch_{epoch}_{mean(loss_o)}.pt')
