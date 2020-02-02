import sys, os
import os.path as osp
import numpy as np
from metrics import fid_score
from metrics import metric_utils as utils
import h5py as hf
import torch
from foundation import train, util
from foundation.train import datasets
from hybrid import get_model
from torch.nn.functional import adaptive_avg_pool2d
import argparse
import pickle as pkl
from tqdm import tqdm

import foundation as fd
from foundation import util
from foundation import train as trn

import hybrid
from run_fid import compute_inception_stat
# from analyze


def get_parser():
	parser = argparse.ArgumentParser(description='Evaluate FID')
	parser.add_argument('--config', type=str)
	# parser.add_argument('--model', type=str, required=True, metavar='M',
	#                     help='path to saved model files to be evaluated')
	parser.add_argument('--save-path', type=str, default=None, metavar='N',
	                    help='Path to save result files')
	# parser.add_argument('--split', type=str, default='all', metavar='S',
	#                     help='Split to eval Fid on (train/test/full/ all)')
	parser.add_argument('--batch_size', type=int, default=64, metavar='B',
	                    help='Batch size')
	parser.add_argument('--seed', type=int, default=0, metavar='S',
	                    help='random seed (default: 0)')
	# parser.add_argument('--save-stat', action='store_true', help='Genrerate and save statistics')
	parser.add_argument('--n-samples', type=int, default=50000, metavar='NS',
	                    help='Number of samples to evaluate/train statistics on')
	parser.add_argument('--pbar', action='store_true')
	return parser

class Dataset_Generator(object):
	def __init__(self, dataset):
		self.dataset = dataset
		pass

	def __call__(self, N):

		idx = torch.randint(0, len(self.dataset), size=(N,))

		if isinstance(self.dataset, fd.data.Batchable_Dataset):
			imgs = self.dataset[idx][0]
		else:
			imgs = []
			for i in idx:
				imgs.append(self.dataset[i][0])

			imgs = torch.stack(imgs)

		return imgs.to('cuda')


def main(argv=None):

	if sys.gettrace() is not None:
		print('in pycharm')

		c = 'n/mpi3d'
		c = 'n/celeba'
		name = 'mpi3d_stats_fid.pkl'

		argv = '--pbar --save-path /is/ei/fleeb/workspace/local_data/mpi3d/{} --config {}'.format(name, c).split(' ')

		# argv.extend(['--n-samples', '100'])

		return 1 # no accidental runs in debugger

	else:
		print('not in pycharm')


	parser = get_parser()
	args = parser.parse_args(argv)

	root = '/is/ei/fleeb/workspace/local_data/fid_stats'
	name = 'mpi3d_real_stats_fid.pkl'

	args.save_path = os.path.join(root, name)
	print('using {}'.format(args.save_path))

	args.config = 'n/mpi3d'
	print('using {}'.format(args.config))

	args.pbar = True

	print(args)

	util.set_seed(args.seed)
	print('Set seed: {}'.format(args.seed))

	C = trn.get_config(args.config)

	C.begin()
	C.dataset.train = False
	# C.dataset.category = 'real'

	dataset = trn.get_dataset(info=C.dataset)
	C.abort()

	print('Loaded dataset: {}'.format(len(dataset)))

	gen = Dataset_Generator(dataset)

	util.set_seed(args.seed)
	print('Set seed: {}'.format(args.seed))

	m, s = compute_inception_stat(gen, batch_size=args.batch_size, n_samples=args.n_samples,
	                              pbar=tqdm if args.pbar else None)

	print(m.shape, s.shape)

	pkl.dump({'m':m, 'sigma':s}, open(args.save_path, 'wb'))

	print('Saved stats to {}'.format(args.save_path))



if __name__ == '__main__':
	sys.exit(main())




