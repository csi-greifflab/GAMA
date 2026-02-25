"""
Iterating over results folder and validating the existence of all 270
experimental results. Then iterating over all 270 results and computing
the GAMA metric between the initializer model and the trained model
"""

import os


import torch
import pandas as pd
import numpy as np

def find_files(path, tag):
	file_paths = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if tag in file:
				path = os.path.join(root,file)
				file_paths.append(path)
	return file_paths


def find_dirs(path, tag):
	dir_paths = []
	for root, dirs, files in os.walk(path):
		for dir in dirs:
			if tag in dir:
				path = os.path.join(root, dir)
				dir_paths.append(path)
	return dir_paths


inpu_file = find_files('.', '.pt')
inpu_csv = find_files('.', '.csv')
lst_csv = [i for i in inpu_csv if '/new' not in i and 'training_sequences.csv' in i]


f = [i for i in inpu_file if 'id' in i]
f2 = [i for i in f if '/models' not in i]
f3 = [i for i in f2 if 'ig_mass' not in i]


empt_lst = []
corrupt_lst = []
good_lst = []
for j in range(270):
	test = [i for i in f3 if f'/id_{j}__' in i]
	if len(test)!=2:
		empt_lst.append(j)
		print(j, len(test))
	else:
		good_count = 0
		for k in test:
			try:
				results = torch.load(k, map_location='cpu')
				print(j, results.shape)
			except Exception:
				corrupt_lst.append(j)
				print('corrupt:',j)
			else:
				good_count += 1
		if good_count == 2:
			good_lst.append(j)




SEQU_LEN = 16
DIM = SEQU_LEN + 2
results = np.zeros((270, DIM, DIM))

file_lst = lst_csv + f3
for index in range(270):
	file_index_lst = 0
	if index not in good_lst:
		continue

	# load file with index
	file_index_lst = [i for i in file_lst if f'/id_{index}__' in i]
	for file in file_index_lst:
		if 'ig_matrix_Init_model.pt' in file:
			x_baseline = torch.load(file, map_location='cpu')
			continue
		if 'training_sequences.csv' in file:
			df_seq_baseline = df_seq_13 = pd.read_csv(file)
			continue
		if 'lstm' in file:
			x_13 = torch.load(file, map_location='cpu')
			continue
		print('FAILED!!!', index)

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

	# model wise normalization
	df.loc[df['10/12'] == '1&3_base', 'integrated gradients'] -= df.loc[df['10/12'] == '1&3_base', 'integrated gradients'].mean()
	df.loc[df['10/12'] == '1&3_base', 'integrated gradients'] /= df.loc[df['10/12'] == '1&3_base', 'integrated gradients'].std()
	df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'] -= df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'].mean()
	df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'] /= df.loc[df['10/12'] == '1&3_model13', 'integrated gradients'].std()

	def compute_rv(df_l, filter_p, filter_q, outdim, sequpos):
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
			rp_np[i, j] = compute_rv(df_l=df, filter_p='1&3_base', filter_q='1&3_model13', outdim=i, sequpos=j)
			results[index, :, :] = rp_np

torch.save(results, 'evalExpAll_seed0.pt')
