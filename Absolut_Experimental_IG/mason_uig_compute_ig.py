"""
The script computes the IG-hypercube as described in the paper on an LSTM and
CDRH3 sequences of HER-2 binders.
"""

import os
import sys

import torch
import pandas as pd

sys.path.append('..')
from ig_lstm import LSTMModel

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
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
reverse_encoding = dict(enumerate(ALPHABET, start=1))
encoding = dict(zip(reverse_encoding.values(), reverse_encoding.keys()))
train_list = [torch.LongTensor([encoding[j] for j in i]).to(DEVICE) for i in df.AASeq.tolist()]


# init model
model = LSTMModel(1024, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(PATH_MODEL, map_location=DEVICE))

input_pos_dim = len(train_list[0])
ENCODING_DIM = 22
output_pos_dim = len(train_list[0])
DIFF_AA = 22
hycube_ig = torch.zeros((input_pos_dim, ENCODING_DIM, output_pos_dim, DIFF_AA, len(train_list)))
for sequ_idx, sequ in enumerate(train_list):# one squence ca. 1:50min
    print(sequ_idx)
    if sequ_idx % 10 == 0:
        if sequ_idx % 20 == 0:
            torch.save(hycube_ig, f'even{PATH_INTERSAVE_HYPERCUBE}')
        else:
            torch.save(hycube_ig, f'uneven{PATH_INTERSAVE_HYPERCUBE}')

    hycube_ig[:, :, :, :, sequ_idx] = model.ig_sample_spc(peptide=sequ, output_pos_dim=output_pos_dim, diff_aa=DIFF_AA)

torch.save(hycube_ig, PATH_SAVE_HYPERCUBE)
