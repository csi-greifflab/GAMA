import time
import sys
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

from ig_single_exp import experiment_run

def gpu_filler(experiment_i = 0, results_folder = '.'):
	# arg_dict = list()
	# arg_dict.append({
	# 'logic_op':'AND',
	# 'sequence_length':5,
	# 'signal_pos':(1, 3),
	# 'signal_sequences_n':5_000,
	# 'ig_sequences_n':150,
	# 'signal2noise':1,
	# 'DEVICE':'cuda:0',
	# 'prj_path':'./t1'
	# })

	# arg_dict.append({
	# 'logic_op':'AND',
	# 'sequence_length':5,
	# 'signal_pos':(1, 3),
	# 'signal_sequences_n':5_000,
	# 'ig_sequences_n':150,
	# 'signal2noise':0.9,
	# 'DEVICE':'cuda:1',
	# 'prj_path':'./t2'
	# })

	# arg_dict.append({
	# 'logic_op':'AND',
	# 'sequence_length':5,
	# 'signal_pos':(1, 3),
	# 'signal_sequences_n':10_000,
	# 'ig_sequences_n':150,
	# 'signal2noise':0.8,
	# 'DEVICE':'cuda:1',
	# 'prj_path':'./t3'
	# })

	# start = time.time()
	# experiment_run(**arg_dict[element])
	# end = time.time()


	# start = time.time()
	# for exp in arg_dict:
	# 	experiment_run(**exp)
	# end = time.time()

	# start = time.time()
	# with ProcessPoolExecutor(max_workers=2) as executor:
	# 	for exp in arg_dict:
	# 		executor.submit(experiment_run, **exp)
	# 		print(exp)
	# 	print('all submited')
	# end = time.time()


	start = time.time()
	arg_dict = list()
	# 270 experimental conditions
	for log_string in ['AND', 'OR', 'XOR']:
		for s2n in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
			for signal_pos in [(2, 4), (7, 9), (13, 15),\
							   (2, 4, 6), (7, 9, 11), (12, 14, 16),\
							   (2, 3, 4, 5), (6, 7, 8, 9), (11, 12, 13, 14)]:
				arg_dict.append({
				'logic_op':log_string,
				'sequence_length':16,
				'signal_pos':signal_pos,
				'signal_sequences_n':10_000,
				'ig_sequences_n':1000,
				'signal2noise':s2n,
				'DEVICE':'cuda:0'
				})
	exp_tmp = arg_dict[experiment_i]
	path = f"{results_folder}/id_{experiment_i}__logic_op_{exp_tmp['logic_op']}__signal_pos_{reduce(lambda a, b: str(a)+str(b), exp_tmp['signal_pos'] )}__signal2noise_{exp_tmp['signal2noise']}"
	exp_tmp['prj_path'] = path
	os.makedirs(path)
	experiment_run(**exp_tmp)
	end = time.time()

	print(f'resources children: {resource.getrusage(resource.RUSAGE_CHILDREN)}')
	print(f'resources self: {resource.getrusage(resource.RUSAGE_SELF)}')
	print(f'time elapsed (paralel): {end - start}')

if __name__ == '__main__':
	gpu_filler(experiment_i=int(sys.argv[1]), results_folder=str(sys.argv[2]))

