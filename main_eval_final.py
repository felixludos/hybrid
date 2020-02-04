import sys, os, time
import shutil
import argparse
import traceback
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
from foundation import train as trn

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

import pickle

from analyze import *

import evaluate as dis_eval
from hybrid import get_model, get_data

from run_fid import compute_inception_stat, load_inception_model, compute_frechet_distance

# from tqdm import tqdm_notebook as tqdm

# tqdm = None

# def get_parser(parser=None):
#
# 	if parser is None:
# 		parser = argparse.ArgumentParser(description='Evaluate Models based on some filter')
#
# 	parser.add_argument('--dataroot', type=str, default=None)
# 	parser.add_argument('--save-dir', type=str, default=None)
#
#
# 	parser.add_argument('--names', type=str, nargs='+', default=None)
#
#
# 	parser.add_argument('--include', type=str, nargs='+', default=[])
#
# 	# filter
# 	parser.add_argument('--remove-all', action='store_true')
# 	parser.add_argument('--jobs', type=int, nargs='+', default=None)
# 	parser.add_argument('--models', type=str, nargs='+', default=None)
# 	parser.add_argument('--datasets', type=str, nargs='+', default=None)
# 	parser.add_argument('--strs', type=str, nargs='+', default=None)
# 	parser.add_argument('--min-ckpt', type=int, default=None)
#
#
#
# 	parser.add_argument('--skip', type=int, default=None)
# 	parser.add_argument('--auto-skip', action='store_true')
#
#
# 	parser.add_argument('--ckpt', type=int, default=None)
# 	parser.add_argument('--overwrite', action='store_true')
# 	parser.add_argument('--pbar', action='store_true')
#
# 	return parser


def gen_prior(model, N):
	return model.generate(N)


def gen_target(model, X=None, Q=None, hybrid=False, ret_q=False):
	if Q is None:
		assert X is not None

		Q = model.encode(X)
		if isinstance(Q, distrib.Distribution):
			Q = Q.mean

	if hybrid:
		Q = util.shuffle_dim(Q)

	gen = model.decode(Q)

	if ret_q:
		return gen, Q
	return gen


def _new_loader(dataset, batch_size, shuffle=False):
	return trn.get_loaders(dataset, batch_size=batch_size, shuffle=shuffle)


def gen_batch(dataset, N=None, loader=None, shuffle=False, seed=None, ret_loader=False):
	if seed is not None:
		util.set_seed(seed)

	if loader is None:
		assert N is not None
		loader = iter(_new_loader(dataset, batch_size=N, shuffle=shuffle))

	try:
		batch = util.to(next(loader), 'cuda')
		B = batch.size(0)
	except StopIteration:
		pass
	else:
		if N is None or B == N:
			if ret_loader:
				return batch[0], loader
			return batch[0]

	loader = iter(_new_loader(dataset, batch_size=N, shuffle=shuffle))
	batch = util.to(next(loader), 'cuda')

	if ret_loader:
		return batch[0], loader

	return batch[0]


def compute_all_fid_scores(model, dataset, fid_stats_ref_path, fid=None):
	if fid is None:
		fid = {'scores': {}, 'stats': {}}

	path = os.path.join(os.environ["FOUNDATION_DATA_DIR"], fid_stats_ref_path)
	f = pickle.load(open(path, 'rb'))
	ref_stats = f['m'][:], f['sigma'][:]

	inception = load_inception_model(dim=2048, device='cuda')

	n_samples = 50000
	# n_samples = 100 # testing

	# rec
	name = 'rec'
	if name not in fid['scores']:
		util.set_seed(0)
		loader = None

		def _generate(N):
			nonlocal loader
			X, loader = gen_batch(dataset, loader=loader, shuffle=True, N=N, ret_loader=True)
			return gen_target(model, X=X, hybrid=False)

		stats = compute_inception_stat(_generate, inception=inception, pbar=tqdm, n_samples=n_samples)

		fid['scores'][name] = compute_frechet_distance(*stats, *ref_stats)
		fid['stats'][name] = stats
	print('FID-rec: {:.2f}'.format(fid['scores'][name]))

	# hyb
	name = 'hyb'
	if name not in fid['scores']:
		util.set_seed(0)
		loader = None

		def _generate(N):
			nonlocal loader
			X, loader = gen_batch(dataset, loader=loader, shuffle=True, N=N, ret_loader=True)
			return gen_target(model, X=X, hybrid=True)

		stats = compute_inception_stat(_generate, inception=inception, pbar=tqdm, n_samples=n_samples)

		fid['scores'][name] = compute_frechet_distance(*stats, *ref_stats)
		fid['stats'][name] = stats
	print('FID-hybrid: {:.2f}'.format(fid['scores'][name]))

	# prior
	name = 'prior'
	if name not in fid['scores']:
		util.set_seed(0)

		def _generate(N):
			return gen_prior(model, N)

		stats = compute_inception_stat(_generate, inception=inception, pbar=tqdm, n_samples=n_samples)

		fid['scores'][name] = compute_frechet_distance(*stats, *ref_stats)
		fid['stats'][name] = stats
	print('FID-prior: {:.2f}'.format(fid['scores'][name]))

	return fid


_disent_eval_fns = {
	'IRS': dis_eval.eval_irs,
	    'MIG': dis_eval.eval_mig, # testing
	    'DCI': dis_eval.eval_dci,
	    'SAP': dis_eval.eval_sap,
	    'ModExp': dis_eval.eval_modularity_explicitness,
	    'Unsup': dis_eval.eval_unsupervised,

	    'bVAE': dis_eval.eval_beta_vae,
	    'FVAE': dis_eval.eval_factor_vae,
}


def compute_all_disentanglement(model, disent=None):
	if disent is None:
		disent = {}

	dataset = dis_eval.shapes3d.Shapes3D()
	repr_fn = dis_eval.representation_func(model, 'cuda')

	itr = tqdm(_disent_eval_fns.items(), total=len(_disent_eval_fns))

	for name, eval_fn in itr:
		itr.set_description(name)
		if name not in disent:
			disent[name] = eval_fn(model='', representation_function=repr_fn, dataset=dataset, seed=0)
		print('{}: {}'.format(name, disent[name]))

	return disent


def _full_analyze(run, save_dir):
	# def _full_analyze(run):

	S = run.reset()

	dname = run.meta.dataset

	if 'box' in dname:
		dname = '3dshapes'

	# check for existing results

	save_path = os.path.join(save_dir, run.name)
	util.create_dir(save_path)

	results_path = os.path.join(save_path, 'results.pth.tar')
	evals_path = os.path.join(save_path, 'evals.pth.tar')

	if os.path.isfile(results_path):
		print('Found existing results: {}'.format(results_path))
		results = torch.load(results_path)
		print(results.keys())
	else:
		results = {}

	if os.path.isfile(evals_path):
		print('Found existing evals: {}'.format(evals_path))
		evals = torch.load(evals_path)
		print(evals.keys())
	else:
		evals = {}

	# check for completion
	if 'fid' in results and ('disent' in results or dname != '3dshapes') and 'H' in results:
		print('Skipping {}, all analysis is already done'.format(run.name))
		# raise Exception()  # testing
		return

	# build dataset

	print('loading dataset {} for {}'.format(run.meta.dataset, run.name))

	eval_disentanglement_metrics = dname == '3dshapes'

	if dname == '3dshapes':

		C = trn.get_config('n/3dshapes')
		fid_stats_ref_path = '3dshapes/3dshapes_stats_fid.pkl'
		#     batch_size = 128

		pass
	elif dname in 'celeba':
		C = trn.get_config('n/celeba')
		fid_stats_ref_path = 'celeba/celeba_stats_fid.pkl'
		fid_stats = ''
	#     batch_size = 32
	elif dname == 'atari' or dname in {'spaceinv', 'pacman', 'seaquest', 'asterix'}:
		C = trn.get_config('n/atari')
		#     batch_size = 32

		C.dataset.game = run.config.dataset.game
		print('using {} game'.format(C.dataset.game))
		fid_stats_ref_path = 'fid_stats/{}_fid_stats.pkl'.format(C.dataset.game)
	# get game

	elif dname == 'mpi3d':
		C = trn.get_config('n/mpi3d')

		C.dataset.category = run.config.dataset.category
		print('using {} cat'.format(C.dataset.category))
		fid_stats_ref_path = 'mpi3d/mpi3d_{}_stats_fid.pkl'.format(C.dataset.category)

	#     batch_size = 128

	# get category

	else:
		raise Exception('{} not found'.format(dname))

	batch_size = 128

	C.dataset.device = 'cpu'

	C.dataset.train = False

	if 'val_split' in C.dataset:
		del C.dataset.val_split

	print('loading model {}'.format(run.ckpt_path))

	S.A = trn.get_config()

	if run.meta.dataset in {'celeba', 'atari', 'spaceinv', 'pacman', 'seaquest', 'asterix'}:
		din, dout = (3, 128, 128), (3, 128, 128)
	else:
		din, dout = (3, 64, 64), (3, 64, 64)

	S.A.din, S.A.dout = din, dout

	with redirect_stdout(open('/dev/null', 'w')):
		run.load(fast=True)
	#     run.load(fast=True)

	model = S.model

	print('model loaded')


	if dname == '3dshapes':  # disentanglement metrics

		if 'disent' not in results:
			results['disent'] = {}

		print('Computing Disentanglement metrics')

		results['disent'] = compute_all_disentanglement(model, disent=results['disent'])
		torch.save(results, results_path)
		print('Saved results to {}'.format(results_path))
		evals['disent'] = results['disent']

	datasets, = trn.load(A=C, get_data=get_data, get_model=None, mode='test')
	dataset = datasets[0]

	S.dataset = dataset
	S.dname = dname

	S.batch_size = batch_size

	print('dataset {} loaded: {}'.format(run.meta.dataset, len(dataset)))

	# run

	# rec error

	if 'L' not in results:
		print('Computing rec loss')

		dataset = S.dataset
		batch_size = S.batch_size

		util.set_seed(1)
		loader = _new_loader(dataset, batch_size=batch_size, shuffle=True)

		loader = tqdm(loader)
		loader.set_description('Evaluating rec error')

		criterion = nn.BCELoss(reduction='none')

		L = []
		Q = None
		R = None
		O = None

		with torch.no_grad():

			for batch in loader:

				batch = util.to(batch, 'cuda')
				X = batch[0]
				B = X.size(0)

				rec = gen_target(model, X=X, hybrid=False, ret_q=Q is None)
				if Q is None:
					rec, Q = rec
					Q = Q.cpu()
				elif R is None:
					R = rec.cpu()
					O = X.cpu()

				loss = criterion(rec, X).view(B, -1).sum(-1)
				L.append(loss)

				# if R is not None:
				# 	break  # TESTING

			util.set_seed(2)
			G = model.generate(len(R)).cpu()
			util.set_seed(2)
			H = gen_target(model, Q=Q.cuda(), hybrid=True).cpu()

			L = torch.cat(L)

		del loader

		results.update({
			'O': O,  # original images
			'R': R,  # reconstructed images
			'L': L,  # reconstruction loss
			'Q': Q,  # latent vectors
			'G': G,  # generated samples using prior
			'H': H,  # generated samples using hybridization (drop-in, prob=1)
			'key': {
				'O': 'original images',
				'R': 'reconstructed images',
				'L': 'reconstruction error of each sample in the test set',
				'Q': 'latent vectors',
				'G': 'images generated from the prior',
				'H': 'images generated using hybridization (dropin, prob=1)',
			}
		})

	# FID score

	if 'fid' not in results:
		results['fid'] = {}
	if 'scores' not in results['fid']:
		results['fid']['scores'] = {}
		results['fid']['stats'] = {}

	if len(results['fid']['scores']) < 3:
		print('Computing FID scores')

		with torch.no_grad():
			results['fid'] = compute_all_fid_scores(model, dataset, fid_stats_ref_path, fid=results['fid'])

		torch.save(results, results_path)
		print('Saved results to {}'.format(results_path))

	if 'fid' not in evals:
		evals['fid'] = results['fid']['scores']

	print('Run {} complete'.format(run.name))

	run.state.evals = evals
	run.state.results = results

	run.save(save_dir=save_dir, overwrite=True, )

	run.reset()


def main(argv=None):

	# if sys.gettrace() is not None:
	# 	argv = [ 'spaceinv-wae_0064-6050849-01_200202-190020', '/is/ei/fleeb/workspace/chome/results/final']

	if argv is None:
		argv = sys.argv[1:]


	print(argv)


	run_name = argv[0]

	save_dir = argv[1]

	print('run: {}'.format(run_name))
	print('save_dir: {}'.format(save_dir))

	# os.environ['FOUNDATION_SAVE_DIR'] = '/is/ei/fleeb/workspace/chome/trained_nets' # testing
	# os.environ['FOUNDATION_DATA_DIR'] = '/is/ei/fleeb/workspace/chome/local_data'


	M = Hybrid_Controller().filter_strs('!test')
	M.clear()
	M.add(run_name)
	M.prep_info()

	assert len(M) == 1

	run = M[0]

	_full_analyze(run, save_dir)

	print('Evaluation complete!')


if __name__ == '__main__':
	sys.exit(main())
