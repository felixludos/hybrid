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



import evaluate as dis_eval
from hybrid import get_model, get_data

from run_fid import compute_inception_stat, load_inception_model, compute_frechet_distance

from analyze import *

# from tqdm import tqdm_notebook as tqdm

# tqdm = None


def save_imgs(imgs, root, name_fmt='{}.png'):
	imgs = imgs.cpu().permute(0, 2, 3, 1).mul(255).byte().numpy()
	for i, img in enumerate(imgs):
		Image.fromarray(img).save(os.path.join(root, name_fmt.format(i)))
	return root

def _full_viz(run):

	# idx = 0
	batch = 16
	steps = 20
	border, between = 0.02, 0.01
	# scale = 2
	# fps = 3
	pbar = tqdm

	save_dir = os.path.join(run.path, 'viz')
	util.create_dir(save_dir)
	existing = set(os.listdir(save_dir))

	run.results = torch.load(os.path.join(run.path, 'results.pth.tar'))

	print('results loaded')


	O = run.results['O']
	H = run.results['H']
	G = run.results['G']
	R = run.results['R']
	Q = run.results['Q']
	# Q.shape, O.shape

	large_dataset = False
	if O.shape[2] == 128:
		print('dealing with a large dataset')
		# batch = 8
		large_dataset = True
		Q = Q[:64]
		O = O[:64]


	if 'traversals' not in existing or 'intervention_diffs' not in run.results:

		S = run.reset()
		S.A = trn.get_config()
		din = O.shape[1:]
		S.A.din, S.A.dout = din, din

		print('loading model')

		with redirect_stdout(open('/dev/null', 'w')):
			run.load(fast=True)

		print('Model loaded')
		model = S.model

		with torch.no_grad():
			_, Q = gen_target(model, X=O.cuda(), ret_q=True)
		Q = Q.cpu()

	else:
		print('No need to load the model')
		model = None



	vecs = None
	walks = None
	deltas = None
	diffs = None

	def _get_walks(lim=None):
		nonlocal walks, vecs, diffs, deltas
		if walks is not None:
			return walks if lim is None else walks[:lim]
		assert model is not None
		vecs = get_traversal_vecs(Q[:lim], steps=steps,
		                          mnmx=(Q.min(0)[0].unsqueeze(-1), Q.max(0)[0].unsqueeze(-1))).contiguous()
		deltas = torch.diagonal(vecs, dim1=-3, dim2=-1)
		walks = get_traversals(vecs, model)
		return walks

	def _get_diffs():
		nonlocal walks, vecs, diffs, save_dir
		if diffs is not None:
			return diffs
		if 'intervention_diffs' not in run.results:
			if walks is None:
				walks = _get_walks()
			run.results['intervention_diffs'] = compute_diffs(walks)
			run.results['active_dims'] = (run.results['intervention_diffs'].mean(0) > 1 / 255 / 2).sum().item()

			# save run.results
			torch.save(run.results, os.path.join(run.path, 'results.pth.tar'))

		return run.results['intervention_diffs']

	name = 'original'
	if name not in existing:
		imgs = O[:batch]
		imgdir = os.path.join(save_dir, name)
		util.create_dir(imgdir)
		print('saving {} to: {}'.format(name, imgdir))
		save_imgs(imgs, imgdir, 'orig{}.png')
		fig = show_nums(imgs, figsize=(9, 9))
		plt.subplots_adjust(wspace=between, hspace=between,
		                    left=border, right=1 - border, bottom=border, top=1 - border)
		fig.savefig(os.path.join(save_dir, 'originals.png'))
		print('{} saved'.format(name))

	name = 'hybrid'
	if name not in existing:
		imgs = H[:batch]
		imgdir = os.path.join(save_dir, name)
		util.create_dir(imgdir)
		print('saving {} to: {}'.format(name, imgdir))
		save_imgs(imgs, imgdir, 'hyb{}.png')
		fig = show_nums(imgs, figsize=(9, 9))
		plt.subplots_adjust(wspace=between, hspace=between,
		                    left=border, right=1 - border, bottom=border, top=1 - border)
		fig.savefig(os.path.join(save_dir, 'hybrids.png'))
		print('{} saved'.format(name))

	name = 'gen'
	if name not in existing:
		imgs = G[:batch]
		imgdir = os.path.join(save_dir, name)
		util.create_dir(imgdir)
		print('saving {} to: {}'.format(name, imgdir))
		save_imgs(imgs, imgdir, 'gen{}.png')
		fig = show_nums(imgs, figsize=(9, 9))
		plt.subplots_adjust(wspace=between, hspace=between,
		                    left=border, right=1 - border, bottom=border, top=1 - border)
		fig.savefig(os.path.join(save_dir, 'gen.png'))
		print('{} saved'.format(name))

	name = 'recs'
	if name not in existing:
		imgs = R[:batch]
		imgdir = os.path.join(save_dir, name)
		util.create_dir(imgdir)
		print('saving {} to: {}'.format(name, imgdir))
		save_imgs(imgs, imgdir, 'rec{}.png')
		fig = show_nums(imgs, figsize=(9, 9))
		plt.subplots_adjust(wspace=between, hspace=between,
		                    left=border, right=1 - border, bottom=border, top=1 - border)
		fig.savefig(os.path.join(save_dir, 'recs.png'))
		print('{} saved'.format(name))

	if 'latent.png' not in existing:
		print('saving latent distributions')
		fig, _ = viz_latent(Q)
		fig.savefig(os.path.join(save_dir, 'latent.png'))
		print('latent saved to {}'.format(os.path.join(save_dir, 'latent.png')))

	if 'interventions.png' not in existing:
		diffs = _get_diffs()
		print('saving interventions')
		fig, _ = viz_interventions(diffs)
		fig.savefig(os.path.join(save_dir, 'interventions.png'))
		print('interventions saved to {}'.format(os.path.join(save_dir, 'interventions.png')))
	print('active dims: {}'.format(run.results['active_dims']))

	if 'traversals' not in existing:

		if large_dataset:
			batch = 8

		walks = _get_walks(batch)

		trav_root = os.path.join(save_dir, 'traversals')

		util.create_dir(trav_root)

		itr = enumerate(walks[:batch])
		if pbar is not None:
			itr = pbar(itr, total=batch)

		tvecs = get_traversal_vecs(Q[:batch], steps=64,
		                           mnmx=(Q.min(0)[0].unsqueeze(-1), Q.max(0)[0].unsqueeze(-1))).contiguous()
		tdeltas = torch.diagonal(tvecs, dim1=-3, dim2=-1)
		tiled_walks = get_traversals(tvecs, model)

		for bidx, full in itr:

			walk_dir = os.path.join(trav_root, 'walk{}_frames'.format(bidx))

			util.create_dir(walk_dir)

			for dim, (frames, nums) in enumerate(zip(full, deltas[bidx].T)):
				util.Animation(get_traversal_anim(frames, vals=nums, scale=1, fps=10)).export(
					os.path.join(walk_dir, 'dim{}-frame.png'.format(dim)), fmt='frames')

			full = tiled_walks[bidx]
			tH, tW = util.calc_tiling(len(full))
			N, S, C, H, W = full.shape
			full = full.view(tH, tW, S, C, H, W)
			full = full.permute(2, 3, 0, 4, 1, 5).contiguous().view(S, C, tH * H, tW * W)
			util.Animation(get_traversal_anim(full, vals=None, scale=1, fps=10)).export(
				os.path.join(trav_root, 'walk{}.mp4'.format(bidx)), fmt='mp4', fps=20)

			plt.close('all')

	pass


def main(argv=None):

	if sys.gettrace() is not None:
		argv = [ '3dshapes-fae_0031-6046954-00_200131-132552_ckpt30']#, '/is/ei/fleeb/workspace/chome/results/final']

		argv = ['spaceinv-fvae_0064-6050849-07_200202-190945_ckpt40']

		# return 0
		# argv[1] = '/is/ei/fleeb/workspace/media/hybrid/final/'

	if argv is None:
		argv = sys.argv[1:]


	print(argv)


	run_name = argv[0] if len(argv) > 0 else None

	# save_dir = argv[1]

	# print('run: {}'.format(run_name))
	# print('save_dir: {}'.format(save_dir))

	# os.environ['FOUNDATION_SAVE_DIR'] = '/is/ei/fleeb/workspace/chome/trained_nets' # testing
	# os.environ['FOUNDATION_DATA_DIR'] = '/is/ei/fleeb/workspace/chome/local_data'


	M = Hybrid_Controller('/is/ei/fleeb/workspace/media/hybrid/final/').filter_strs('!test')
	# M.clear()
	# M.add(run_name)
	M.prep_info(name='model.pth.tar')

	# assert len(M) == 1

	failed = []

	active = M.active

	del M

	while len(active):
		run = active.pop()
		print('running: {} ({} remaining)'.format(run.name, len(active)))
		try:
			_full_viz(run)
		except KeyboardInterrupt as e:
			raise e
		except Exception as e:
			# raise e # testing
			failed.append(run.name)

			print('{} failed'.format(run.name))
			traceback.print_exc()
		del run
		plt.close('all')

	print('Evaluation complete!')

	print('Failed runs:')
	print(failed)


if __name__ == '__main__':
	sys.exit(main())
