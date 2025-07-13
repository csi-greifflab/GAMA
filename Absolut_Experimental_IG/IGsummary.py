import os


import torch    
import pandas as pd
import numpy as np

#masonModel = torch.load('uneven500inter_MasonModelUIG.pt', map_location='cpu')
#masonInit = torch.load('uneven500inter_MasonInitUIG.pt', map_location='cpu')

SEQU_LEN = 11
DIM = SEQU_LEN
results = np.zeros((DIM, DIM))
INTER = 200

#df = pd.read_csv("../mHER_H3_AgPos_unique.csv")#[:INTER]
df = pd.read_csv('5KN5_A_A_MascotteSlices_feature.tsv', sep='\t', header=1)
epmask = '1011011000121030001120000000000000000000'

df = df.query('interMaskAGEpitope == @epmask').reset_index(drop=True)


#alphabet = 'ACDEFGHIKLMNPQRSTVWY'
#encoding = {letter:i for i, letter in enumerate(alphabet, start=1)}
#reverse_encoding = {i:letter for i, letter in enumerate(alphabet)}
#train_list = [torch.LongTensor([0] + [encoding[j] for j in i] + [21]) for i in df.Slide.tolist()]
#df_seq_tmp = pd.DataFrame([s.tolist() for s in train_list])

alphabet = 'ACDEFGHIKLMNPQRSTVWY'
encoding = {letter:i for i, letter in enumerate(alphabet)}
reverse_encoding = {i:letter for i, letter in enumerate(alphabet)}
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

def computeRv(df_l, filter_p, filter_q, outdim, sequpos):
    if outdim == 0:
        return 0
    t = df_l[df_l['outDim']==outdim]
    tv = 0
    for i in range(DIM):
        tmp_filt = t[t['sequence position'] == i]
        tv += tmp_filt[tmp_filt['10/12']==filter_p]['integrated gradients'].var(skipna=False)
        tv += tmp_filt[tmp_filt['10/12']==filter_q]['integrated gradients'].var(skipna=False)
        # tv += tmp_filt['integrated gradients'].var(skipna=False)
    tv = tv / (outdim + 1)
    # print(f'{tv=}, {outdim=}, {sequpos=}')
    # tv = t[t['sequence position'].isin(list(range(1, outdim+1)))]['integrated gradients'].var(skipna=False)
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
        rp_np[i, j] = computeRv(df_l=df, filter_p='1&3_base', filter_q='1&3_model13', outdim=i, sequpos=j)

torch.save(rp_np, f'./uig5KN5_ig/GAMA_5KN5_withtoken.pt')
#torch.save(rp_np, f'GAMA_Mason32796.pt')


# local pc
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def show_values(axs, width_mlt=1.5, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / width_mlt
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize='x-large') 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left", fontsize='x-large')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

SUMMARY_FILE = 'GAMA_5KN5.pt'
pos = range(1, 12)
SEQU_LEN = 11
results = torch.load(SUMMARY_FILE, map_location='cpu')
df_val = pd.read_csv('entropylistTopKMascot.tsv', sep='\t')
res_average = np.nanmean(results[1:,:], axis=0, where=~np.ma.masked_invalid(results[1:,:]).mask)
abs5KN5 = [8.044719101123478, 3.2284074255007096, 2.950522716170003, 0.0, 10.320766976062318, 17.414616511968898, 10.998270639960587, 0.0, 8.465373717635664, 2.401455788959443, 8.988485588666236]

spR = stats.spearmanr(res_average, abs5KN5)
res.statistic

res_average = abs5KN5
#rp_np = torch.load('GAMA_Mason17250.pt', map_location='cpu')
#res_average = np.nanmean(rp_np[:,:], axis=0, where=~np.ma.masked_invalid(rp_np[:,:]).mask)#[1:-1]

df_plot = pd.DataFrame(abs5KN5, columns=["binding affinity"])
df_plot['binding affinity'] = df_plot['binding affinity']#*(-1)
#df_plot['position'] = list(range(99, 109))
df_plot['position'] = list(pos)
c = sns.color_palette("light:r", n_colors=11)
plt.figure(figsize=(20,5))
ax = sns.barplot(data=df_plot, x="position", y="binding affinity", palette=c)

ax.set_xlabel("Position",fontsize=18)
ax.set_ylabel("binding affinity",fontsize=18)

ax.set_frame_on(False)
#ax.set(ylim=(0, 6.5))
ax.set(ylim=(0, 21))
#ax.set(ylim=(0, 10))
#ax.set_yticklabels(list(pos), size = 15)
#ax.set_xticklabels(list(pos), size = 15)

show_values(ax, 2)
#plt.savefig('G:/My Drive/work/ig/uig/results_fig/masonGama.png', bbox_inches='tight', dpi=600)
plt.savefig('GAMA_5KN5.png', bbox_inches='tight', dpi=600)
plt.show()