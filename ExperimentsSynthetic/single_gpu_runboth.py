"""
function schedules all 270 synthetic experiments from the paper.
so it can be called with the slurm array scheduler
"""

import sys
import os
from functools import reduce

from ig_single_exp_2crit import experiment_run

def gpu_filler(experiment_i = 0, results_folder = '.'):
	arg_dict = []
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
	print('finished')
if __name__ == '__main__':
	gpu_filler(experiment_i=int(sys.argv[1]), results_folder=str(sys.argv[2]))
