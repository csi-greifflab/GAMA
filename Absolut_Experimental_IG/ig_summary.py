"""
The script compares the IG tensor from a randomly initialized LSTM model
and a trained LSTM model.
"""

import torch
import pandas as pd
import numpy as np

SEQU_LEN = 11
DIM = SEQU_LEN
results = np.zeros((DIM, DIM))

#df = pd.read_csv("../mHER_H3_AgPos_unique.csv")#[:INTER]
df = pd.read_csv('5KN5_A_A_MascotteSlices_feature.tsv', sep='\t', header=1)
EPMASK = '1011011000121030001120000000000000000000'

df = df.query('interMaskAGEpitope == @EPMASK').reset_index(drop=True)


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
reverse_encoding = dict(enumerate(ALPHABET, start=1))
encoding = dict(zip(reverse_encoding.values(), reverse_encoding.keys()))
train_list = [torch.LongTensor([encoding[j] for j in i]) for i in df.Slide.tolist()]
df_seq_tmp = pd.DataFrame([s.tolist() for s in train_list])


x_baseline = torch.load('./uig5KN5_ig/Hypercube_5KN5_init.pt', map_location='cpu')
df_seq_baseline = df_seq_13 = df_seq_tmp
x_13 = torch.load('./uig5KN5_ig/Hypercube_5KN5_data.pt', map_location='cpu')

merged_ten = torch.cat((x_baseline, x_13), dim=4)
merged_csv = pd.concat([df_seq_baseline, df_seq_13], ignore_index=True)
DATA_LENGTH = len(merged_csv)
z = torch.ones(len(merged_csv), DIM, DIM)#(data, input dim, output din)
tmp = merged_ten.mean(dim=3)
for i in range(len(merged_csv)):
    for j in range(DIM):
        z[i, :, j] = tmp[:, merged_csv.loc[i,:].tolist(), j, i].diagonal(dim1=0, dim2=1)

indices = pd.MultiIndex.from_product((range(DATA_LENGTH), range(DIM), range(DIM)), names=('lines', 'sequence position', 'outDim'))
xnp = z.reshape(-1).numpy()
df = pd.DataFrame(xnp, index=indices, columns=('integrated gradients',)).reset_index()
df['10/12'] = ['False'] * (DATA_LENGTH * DIM * DIM)


for i in range(DATA_LENGTH):
    if i < len(df_seq_baseline):
        df.loc[df['lines'] == i, '10/12'] = '1&3_base'
    else:
        df.loc[df['lines'] == i, '10/12'] = '1&3_model13'

# model wise normalization (1&3 model)
df.loc[df['10/12'] == '1&3_base', 'integrated gradients'] -= df.loc[df['10/12'] == '1&3_base', 'integrated gradients'].mean()
df.loc[df['10/12'] == '1&3_base', 'integrated gradients'] /= df.loc[df['10/12'] == '1&3_base', 'integrated gradients'].std()
df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'] -= df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'].mean()
df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'] /= df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'].std()

def compute_rv(df_l, filter_p, filter_q, outdim, sequpos):
    """
    Callculate the median difference between the gradients of the
    initilizer model and the trained model. 

    Args:
    df_l: Pandas dataframe.
    filter_p: The initilizer model key.
    filter_q: The trained model key.
    outdim: Output dimension for which the gradient is callculated.
    sequpos: Input dimension for which the gradient is callculated.

    Returns:
        float: Variance normalized median difference between.
    """
    if outdim == 0:
        return 0
    t = df_l[df_l['outDim']==outdim]
    tv = 0
    for i in range(DIM):
        tmp_filt = t[t['sequence position'] == i]
        tv += tmp_filt[tmp_filt['10/12']==filter_p]['integrated gradients'].var(skipna=False)
        tv += tmp_filt[tmp_filt['10/12']==filter_q]['integrated gradients'].var(skipna=False)
    tv = tv / (outdim + 1)
    t = t[t['sequence position']==sequpos]
    q = t[t['10/12']==filter_p]['integrated gradients']
    p = t[t['10/12']==filter_q]['integrated gradients']
    q_med = q.median(skipna=False)
    p_med = p.median(skipna=False)
    med_diff = abs(p_med-q_med) # med_diff should be high for positions where signal is implanted
    q_var = q.var(skipna=False)
    p_var = p.var(skipna=False)
    rv = (q_var + p_var) / tv # rv should be small for positions where signal is implanted
    return med_diff / rv

rp_np = np.zeros([DIM, DIM])
for i in range(DIM):
    for j in range(DIM):
        rp_np[i, j] = compute_rv(df_l=df, filter_p='1&3_base', filter_q='1&3_model13', outdim=i, sequpos=j)

torch.save(rp_np, './uig5KN5_ig/GAMA_5KN5_withtoken.pt')
#torch.save(rp_np, f'GAMA_Mason32796.pt')
