import sys, os, time
import shutil
import argparse
# %load_ext autoreload
# %autoreload 2
# os.environ['FOUNDATION_RUN_MODE'] = 'jupyter'
# os.environ['FOUNDATION_SAVE_DIR'] = '/is/ei/fleeb/workspace/chome/trained_nets'
# os.environ['FOUNDATION_DATA_DIR'] = '/is/ei/fleeb/workspace/local_data'
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
# from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
import torchvision.models
import torchvision

from torch.utils.data import Dataset, DataLoader, TensorDataset
import gym
import numpy as np
# %matplotlib notebook
# %matplotlib tk
import matplotlib.pyplot as plt
import imageio
# plt.switch_backend('Agg') #('Qt5Agg')
import foundation as fd
from foundation import models
from foundation import util
from foundation import train

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from bisect import bisect_left

from contextlib import nullcontext, redirect_stdout, redirect_stderr

from hybrid import get_model, get_data
from analyze import Hybrid_Controller, update_checkpoint

from tqdm import tqdm

# plt.ioff()

# np.set_printoptions(linewidth=120, suppress=True)

def get_parser(parser=None):

	if parser is None:
		parser = argparse.ArgumentParser(description='Evaluate Models based on some filter')

	parser.add_argument('--dataroot', type=str, default=None)
	parser.add_argument('--save-dir', type=str, default=None)



	# filter
	parser.add_argument('--jobs', type=int, nargs='+', default=None)
	parser.add_argument('--models', type=str, nargs='+', default=None)
	parser.add_argument('--datasets', type=str, nargs='+', default=None)
	parser.add_argument('--strs', type=str, nargs='+', default=None)


	parser.add_argument('--ckpt', type=int, default=None)
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--pbar', action='store_true')

	return parser



def main(argv=None):

	parser = get_parser()

	if sys.gettrace() is not None:
		print('in pycharm')

		argv = ['--jobs', '157']
		argv.extend(['--dataroot', '/is/ei/fleeb/workspace/chome/trained_nets'])

		return 1 # no accidental runs in debugger

	else:
		print('not in pycharm')

	args = parser.parse_args(argv)

	global tqdm
	if not args.pbar:
		print('Not using a progress bar')
		tqdm = None

	if args.dataroot is None:
		assert 'FOUNDATION_DATA_DIR' in os.environ, 'no default data dir found'
		args.dataroot = os.environ['FOUNDATION_SAVE_DIR']
		print('Using dataroot: {}'.format(args.dataroot))

	if args.save_dir is not None:
		print('Will save all results to: {}'.format(args.save_dir))
		if not os.path.isdir(args.save_dir):
			os.makedirs(args.save_dir)
	else:
		print('WARNING: no results will be saved!')

	# frames = torch.rand(100, 64, 64, 3).mul(255).byte().numpy()
	#
	# vid = util.Video(frames)
	#
	# vid.export('test.mp4')


	M = Hybrid_Controller(root=args.dataroot).filter_strs('!test')

	if args.jobs is not None:
		print('Filtering out all jobs except: {}'.format(', '.join(map(str,args.jobs))))
		M.filter_jobs(*args.jobs)

	if args.models is not None:
		print('Filtering out all models except: {}'.format(', '.join(args.models)))
		M.filter_models(*args.models)

	if args.datasets is not None:
		print('Filtering out all datasets except: {}'.format(', '.join(args.datasets)))
		M.filter_datasets(*args.datasets)

	if args.strs is not None:
		print('Filtering out all except containing: {}'.format(', '.join(args.strs)))
		M.filter_datasets(*args.strs)

	print('\nRemaining jobs:')

	M.show()

	print('Loading checkpoint: {}'.format(args.ckpt if args.ckpt is not None else '[last]'))
	M.load_configs(args.ckpt)

	print('\nCheckpoints: ')
	for run in M.active:
		print(run.ckpt_path)

	print('\nUnique properties')

	M.show_unique()


	runs = M.active
	del M.full_info
	del M.active

	div = '-'*50

	for i, run in enumerate(runs):

		print(div)
		print('Evaluating run {}/{}: {}'.format(i+1,len(runs), run.name))

		run.reset()

		with redirect_stdout(open(os.devnull, 'w')): # hide output from loading
			run.load(pbar=tqdm)

		run.run(pbar=tqdm)

		print('----- Model is loaded.')

		print('Evaluating results')
		run.evaluate(pbar=tqdm)

		update_checkpoint(run.state, 'evals', overwrite=True)

		if args.save_dir is not None:
			print('Visualizing results')
			run.visualize(pbar=tqdm)

			run.save(args.save_dir, overwrite=args.overwrite)

		run.reset() # release memory

		print('\n')


	return 0



if __name__ == '__main__':
	sys.exit(main())
