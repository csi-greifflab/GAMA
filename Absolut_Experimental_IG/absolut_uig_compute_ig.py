"""
The script computes the IG-hypercube as described in the paper on an LSTM and
Absolut! CDRH3 sequences, filtered for a specific epitope interaction mask.
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

PATH_MODEL = './uig5KN5_ig/lstm_epoch_102668_0.6914815171771392.pt'
#PATH_INTERSAVE_HYPERCUBE = './uig3Q3G/Hypercube_3Q3G_inter.pt'
PATH_SAVE_HYPERCUBE = './uig5KN5_ig/full/Hypercube_5KN5_modelLong.pt'
PATH_DATA = '5KN5_A_A_MascotteSlices_feature.tsv'


df = pd.read_csv(PATH_DATA, sep='\t', header=1)
EPMASK = '1011011000121030001120000000000000000000'
df = df.query('interMaskAGEpitope == @EPMASK').reset_index(drop=True)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
reverse_encoding = dict(enumerate(ALPHABET, start=1))
encoding = dict(zip(reverse_encoding.values(), reverse_encoding.keys()))
train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]).to(DEVICE) for i in df.Slide.tolist()]


# init model
model = LSTMModel(1024, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(PATH_MODEL, map_location=DEVICE))

input_pos_dim = len(train_list[0])

ENCODING_DIM = 22
output_pos_dim = len(train_list[0])
DIFF_AA = 22
hycube_ig = torch.zeros((input_pos_dim, ENCODING_DIM, output_pos_dim, DIFF_AA, len(train_list)))
total_len = len(train_list)
for sequ_idx, sequ in enumerate(train_list):# one squence ca. 1:50min
    print(sequ_idx, total_len)
    hycube_ig[:, :, :, :, sequ_idx] = model.ig_sample_spc(peptide=sequ, output_pos_dim=output_pos_dim, diff_aa=DIFF_AA)

torch.save(hycube_ig, PATH_SAVE_HYPERCUBE)
