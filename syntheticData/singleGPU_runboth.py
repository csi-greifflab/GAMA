import time
import sys
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

import torch

from ig_single_exp_2crit import experiment_run

def gpu_filler(experiment_i = 0, results_folder = '.'):
	# arg_dict = list()
	# arg_dict.append({
	# 'logic_op':'AND',
	# 'sequence_length':10,
	# 'signal_pos':(1, 3),
	# 'signal_sequences_n':5_000,
	# 'ig_sequences_n':150,
	# 'signal2noise':1,
	# 'DEVICE':'cuda:1',
	# #'prj_path':'./tmp'
	# })
	# exp_tmp = arg_dict[0]
	# results_folder = './tmp'
	# experiment_i = -2
	# In [4]: arg_dict[89]
	# Out[4]:
	# {'logic_op': 'AND',
	#  'sequence_length': 16,
	#  'signal_pos': (11, 12, 13, 14),
	#  'signal_sequences_n': 10000,
	#  'ig_sequences_n': 1000,
	#  'signal2noise': 0.1,
	#  'DEVICE': 'cuda:0'}


	# time single run 607238,1042597294s approx. 7D
	# 3.4gb

	#start = time.time()
	arg_dict = list()
	# 270 experimental conditions
	# 110 995 computation units, assuming 24h per experiment
	# 
	for log_string in ['AND', 'OR', 'XOR']:
		for signal_pos in [(2, 4), (7, 9), (13, 15),\
						   (2, 4, 6), (7, 9, 11), (12, 14, 16),\
						   (2, 3, 4, 5), (6, 7, 8, 9), (11, 12, 13, 14)]:
			for s2n in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
				arg_dict.append({
				'logic_op':log_string,
				'sequence_length':16,#16
				'signal_pos':signal_pos,
				'signal_sequences_n':10_000,#10_000
				'ig_sequences_n':1000,#1000
				'signal2noise':s2n,
				'DEVICE':'cuda:0',
				'seed_manual':1
				})
	exp_tmp = arg_dict[experiment_i]
	path = f"{results_folder}/id_{experiment_i}__logic_op_{exp_tmp['logic_op']}__signal_pos_{reduce(lambda a, b: str(a)+str(b), exp_tmp['signal_pos'] )}__signal2noise_{exp_tmp['signal2noise']}"
	exp_tmp['prj_path'] = path
	round_one = True
	if os.path.exists(path):
		c1 = os.path.exists(f'{path}/models')
		c2 = os.path.exists(f'{path}/training_sequences.csv')
		c3 = os.path.exists(f'{path}/models/Init_model.pt')
		c4 = len(list(os.scandir(f'{path}/models'))) == 2
		c5 = len(list(os.scandir(f'{path}'))) == 3
		if not (c1 and c2 and c3 and c4 and c5):
			raise RuntimeWarning(f'no valid memory configuration in {experiment_i}')
		round_one = False
	os.makedirs(path, exist_ok=True)
	try:
		experiment_run(**exp_tmp)
	except Exception as e:
		with open(f'{results_folder}/log.txt', "a") as file:
			print(f'{experiment_i}; FAILED: {str(e)}', file=file)
	#end = time.time()
	with open(f'{results_folder}/log.txt', "a") as file:
		if round_one:
			print(f'{experiment_i}; 1. iter finished', file=file)
		else:
			print(f'{experiment_i}; 2. iter finished', file=file)

	round_one = False
	try:
		experiment_run(**exp_tmp)
	except Exception as e:
		with open(f'{results_folder}/log.txt', "a") as file:
			print(f'{experiment_i}; FAILED: {str(e)}', file=file)
	#end = time.time()
	with open(f'{results_folder}/log.txt', "a") as file:
		if round_one:
			print(f'{experiment_i}; 1. iter finished', file=file)
		else:
			print(f'{experiment_i}; 2. iter finished', file=file)
	# with open(f'{path}/log.txt', "w") as file:
	# 	print(f'resources children: {resource.getrusage(resource.RUSAGE_CHILDREN)}', file=file)
	# 	print(f'resources self: {resource.getrusage(resource.RUSAGE_SELF)}', file=file)
	# 	print(f'time elapsed (paralel): {end - start}', file=file)
	# 	print(f'{torch.cuda.get_device_properties(0).total_memory}', file=file)
	# 	print(f'{torch.cuda.mem_get_info(device=0)}', file=file)
	# 	print(f'{torch.cuda.device_count()}', file=file)
	print('finished')
if __name__ == '__main__':
	gpu_filler(experiment_i=int(sys.argv[1]), results_folder=str(sys.argv[2]))

