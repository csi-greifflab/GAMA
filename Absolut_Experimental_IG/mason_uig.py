import os
from statistics import mean

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import pandas as pd

from igLSTM import LSTMModel


DEVICE = 'cuda:1'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#mHER_H3_AgPos_unique.csv
df = pd.read_csv("mHER_H3_AgPos_unique.csv")
alphabet = 'ACDEFGHIKLMNPQRSTVWY'
encoding = {letter:i for i, letter in enumerate(alphabet, start=1)}
reverse_encoding = {i:letter for i, letter in enumerate(alphabet)}
train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]).to(DEVICE) for i in df.AASeq.tolist()]

model = LSTMModel(1024, 22, DEVICE).to(DEVICE)
torch.save(model.state_dict(), f'modelmasonSEToken2_long/Mason_Init_model.pt')
optimizer = optim.Adam(model.parameters())
loss_function = nn.NLLLoss().to(DEVICE)
train_data = DataLoader(train_list, shuffle=True, batch_size=None)

train_loss = pd.DataFrame()

for epoch in range(1_000_000):
    loss_o = list()
    for sentence in train_data:
        model.zero_grad(set_to_none=True)
        tag_scores = model(sentence[0:-1])
        loss = loss_function(tag_scores.view(-1, 22), sentence[1:])
        loss.backward()
        optimizer.step()
        loss_o.append(loss.detach().tolist())
    #train_loss.append(loss_o / len(train_list))
    train_loss[train_loss.shape[1]] = loss_o
    #print(f'{epoch=}, {mean(loss_o)=}')
    print(f'{epoch=}, {mean(loss_o)=}')
    torch.save(model.state_dict(), f'modelmasonSEToken2_long/lstm_epoch_{epoch}_{mean(loss_o)}.pt')



# COMPUTE IG
import os
from statistics import mean

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import pandas as pd

from igLSTM import LSTMModel

DEVICE = 'cuda:0'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PATH_MODEL = 'Mason_Init_model.pt'
PATH_INTERSAVE_HYPERCUBE = 'TestHypercube_inter.pt'
PATH_SAVE_HYPERCUBE = 'TestHypercube.pt'
PATH_DATA = 'mHER_H3_AgPos_unique.csv'


# Load data mHER_H3_AgPos_unique.csv
df = pd.read_csv(PATH_DATA)
alphabet = 'ACDEFGHIKLMNPQRSTVWY'
encoding = {letter:i for i, letter in enumerate(alphabet)}
reverse_encoding = {i:letter for i, letter in enumerate(alphabet)}
train_list = [torch.LongTensor([encoding[j] for j in i]).to(DEVICE) for i in df.AASeq.tolist()]


# init model
model = LSTMModel(1024, 22, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(PATH_MODEL, map_location=DEVICE))

input_pos_dim = len(train_list[0])
encoding_dim = 22
output_pos_dim = len(train_list[0])
diff_aa = 22
hycube_ig = torch.zeros((input_pos_dim, encoding_dim, output_pos_dim, diff_aa, len(train_list)))
for sequ_idx, sequ in enumerate(train_list):# one squence ca. 1:50min
    print(sequ_idx)
    if sequ_idx % 10 == 0:
        if sequ_idx % 20 == 0:
            torch.save(hycube_ig, f'even{PATH_INTERSAVE_HYPERCUBE}')
        else:
            torch.save(hycube_ig, f'uneven{PATH_INTERSAVE_HYPERCUBE}')

    hycube_ig[:, :, :, :, sequ_idx] = model.IG_sample_spc(peptide=sequ, output_pos_dim=output_pos_dim, diff_aa=diff_aa)

torch.save(hycube_ig, PATH_SAVE_HYPERCUBE)