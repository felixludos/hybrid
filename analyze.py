
import sys, os, time
import shutil
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
from IPython import display
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
# %matplotlib tk
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import seaborn as sns
from collections import OrderedDict
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
import foundation as fd
from foundation import models
from foundation import util
from foundation import train

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#from foundation.util import replicate, Cloner

import evaluate as dis_eval
from hybrid import get_model, get_data


def show_nums(imgs, titles=None, H=None, W=None, figsize=(6, 6),
              reverse_rows=False, grdlines=False):
	if H is None and W is None:
		B = imgs.size(0)
		l = int(np.sqrt(B))
		assert l ** 2 == B, 'not right: {} {}'.format(l, B)
		H, W = l, l
	elif H is None:
		H = imgs.shape[0] // W
	elif W is None:
		W = imgs.shape[0] // H

	imgs = imgs.cpu().permute(0, 2, 3, 1).squeeze().numpy()

	fig, axes = plt.subplots(H, W, figsize=figsize)

	if titles is None:
		titles = [None] * len(imgs)

	iH, iW = imgs.shape[1], imgs.shape[2]

	for ax, img, title in zip(axes.flat, imgs, titles):
		plt.sca(ax)
		if reverse_rows:
			img = img[::-1]
		plt.imshow(img)
		if grdlines:
			plt.plot([0, iW], [iH / 2, iH / 2], c='r', lw=.5, ls='--')
			plt.plot([iW / 2, iW / 2], [0, iH], c='r', lw=.5, ls='--')
			plt.xlim(0, iW)
			plt.ylim(0, iH)
		if title is not None:
			plt.xticks([])
			plt.yticks([])
			plt.title(title)
		else:
			plt.axis('off')
	#     fig.tight_layout()
	return fig


def load_fn(S, **unused):

	cpath = S.ckpt_path
	A = S.A if 'A' in S else None

	A, (dataset, *other), model, ckpt = train.load(path=cpath, A=A, get_model=get_model, get_data=get_data,
	                                               return_args=True, return_ckpt=True)


	trainset = dataset
	if len(other) and other[0] is not None:
		print('*** Using validation set')
		dataset = other[0]

	model.eval()

	S.A = A
	S.trainset = trainset
	S.dataset = dataset
	S.other = other
	S.model = model
	S.ckpt = ckpt

	records = ckpt['records']
	print('Trained on {:2.2f} M samples'.format(records['total_samples']['train'] / 1e6))
	if len(other) and other[0] is not None:
		dataset = other[0]
		print('Using validation set')
	else:
		print('Using training set')


	S.records = records

	# DataLoader

	A.dataset.batch_size = 16
	common_Ws = {64: 8, 32: 4, 16: 4, 9: 3, 8: 2, 4: 2}
	border, between = 0.02, 0.01
	img_W = common_Ws[A.dataset.batch_size]
	util.set_seed(0)
	loader = train.get_loaders(dataset, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
	                           shuffle=True, drop_last=False, )
	util.set_seed(0)
	loader = iter(loader)

	S.loader = loader
	S.img_W = img_W
	S.border, S.between = border, between

	# Batch

	batch = next(loader)
	batch = util.to(batch, A.device)

	S.batch = batch
	S.X = batch[0]


def run_model(S, pbar=None, **unused):

	A = S.A
	dataset = S.dataset
	model = S.model
	X = S.X

	with torch.no_grad():

		if model.enc is None:
			q = model.sample_prior(X.size(0))
		else:
			q = model.encode(X)
		qdis = None
		qmle = q
		if isinstance(q, distrib.Distribution):
			qdis = q
			q = q.rsample()
			qmle = qdis.loc
		rec = model.decode(q)
		# vrec = model.disc(rec) if model.disc is not None else None

		p = model.sample_prior(X.size(0))
		gen = model.decode(p)

		h = util.shuffle_dim(q)
		hyb = model.decode(h)

	S.rec = rec
	S.gen = gen
	S.hyb = hyb

	S.q = q
	S.p = p
	S.h = h

	S.qdis = qdis
	S.qmle = qmle

	batch_size = 128  # number of samples to get distribution
	util.set_seed(0)
	int_batch = next(iter(train.get_loaders(dataset, batch_size=batch_size, num_workers=A.num_workers,
	                                        shuffle=True, drop_last=False, )))
	with torch.no_grad():
		int_batch = util.to(int_batch, A.device)
		int_X, = int_batch
		if model.enc is None:
			int_q = model.sample_prior(int_X.size(0))
		else:
			int_q = model.encode(int_X)
		dis_int_q = None
		if isinstance(int_q, distrib.Distribution):
			#         int_q = int_q.rsample()
			dis_int_q = int_q
			int_q = int_q.loc
	del int_batch
	del int_X

	S.int_q = int_q
	S.dis_int_q = dis_int_q


	latent_dim = A.model.latent_dim
	iH, iW = X.shape[-2:]

	rows = 4
	steps = 60
	# bounds = -2,2
	bounds = None
	dlayout = rows, latent_dim // rows
	outs = []

	all_diffs = []
	inds = [0, 2, 3]
	inds = np.arange(len(q))
	save_inds = [0, 1, 2, 3]
	# save_inds =  []

	saved_walks = []

	for idx in inds:

		walks = []
		for dim in range(latent_dim):

			dev = int_q[:, dim].std()
			if bounds is None:
				deltas = torch.linspace(int_q[:, dim].min(), int_q[:, dim].max(), steps)
			else:
				deltas = torch.linspace(bounds[0], bounds[1], steps)
			vecs = torch.stack([int_q[idx]] * steps)
			vecs[:, dim] = deltas

			with torch.no_grad():
				walks.append(model.decode(vecs).cpu())

		walks = torch.stack(walks, 1)
		chn = walks.shape[2]

		dsteps = 10
		diffs = (walks[dsteps:] - walks[:-dsteps]).abs()  # .view(steps-dsteps, latent_dim, chn, 64*64)
		#     diffs /= (walks[dsteps:]).abs()
		# diffs = diffs.clamp(min=1e-10,max=1).abs()
		diffs = diffs.view(steps - dsteps, latent_dim, chn * iH * iW).mean(-1)
		#     diffs = 1 - diffs.mean(-1)
		#     print(diffs.shape)
		#     diffs *= 2
		all_diffs.append(diffs.mean(0))
		#     print(all_diffs[-1])

		if idx in save_inds:
			# save_dir = S.save_dir

			walks_full = walks.view(steps, dlayout[0], dlayout[1], chn, iH, iW) \
				.permute(0, 1, 4, 2, 5, 3).contiguous().view(steps, iH * dlayout[0], iW * dlayout[1], chn).squeeze()
			images = []
			for img in walks_full.cpu().numpy():
				images.append((img * 255).astype(np.uint8))

			saved_walks.append(images)

			# imageio.mimsave(os.path.join(save_dir, 'walks-idx{}.gif'.format(idx, dim)), images)
			#
			# with open(os.path.join(save_dir, 'walks-idx{}.gif'.format(idx, dim)), 'rb') as f:
			# 	outs.append(display.Image(data=f.read(), format='gif'))
			del walks_full

	all_diffs = torch.stack(all_diffs)

	S.all_diffs = all_diffs
	S.saved_walks = saved_walks

	other = S.other

	full_q = None

	if not len(other) or other[0] is None:
		print('No validation set found')

	elif model.enc is not None:

		valset = other[0]

		print('valset: {}'.format(len(valset)))

		valloader = train.get_loaders(valset, batch_size=128, num_workers=A.num_workers,
		                                        shuffle=False, drop_last=False, )

		if pbar is not None:
			valloader = pbar(valloader, total=len(valloader))
			valloader.set_description('Validation set')

		full_q = []

		for batch in valloader:
			batch = util.to(batch, A.device)
			X, = batch

			with torch.no_grad():

				q = model.encode(X)
				if isinstance(q, distrib.Distribution):
					q = torch.stack([q.loc, q.scale], 1)
				full_q.append(q.cpu())

		if len(full_q):
			full_q = torch.cat(full_q)

			print(full_q.shape)

			if len(full_q.shape) > 2:
				full_q = distrib.Normal(loc=full_q[:,0], scale=full_q[:,1])

		else:
			full_q = None

	S.full_q = full_q



def viz_originals(S, **unused):

	X = S.X
	img_W = S.img_W

	fig = show_nums(X, figsize=(9, 9), H=img_W)
	# fig.suptitle("originals", fontsize=14)
	# plt.tight_layout()
	border, between = 0.02, 0.01
	plt.subplots_adjust(wspace=between, hspace=between,
	                    left=border, right=1 - border, bottom=border, top=1 - border)

	return fig,

def viz_reconstructions(S, **unused):

	rec = S.rec
	img_W = S.img_W
	border, between = S.border, S.between

	fig = show_nums(rec, figsize=(9, 9), W=img_W)
	plt.subplots_adjust(wspace=between, hspace=between,
	                    left=border, right=1 - border, bottom=border, top=1 - border)

	return fig,

def viz_hybrids(S, **unused):

	hyb = S.hyb
	img_W = S.img_W
	border, between = S.border, S.between

	fig = show_nums(hyb, figsize=(9, 9), W=img_W)
	plt.subplots_adjust(wspace=between, hspace=between,
	                    left=border, right=1 - border, bottom=border, top=1 - border)

	return fig,

def viz_generated(S, **unused):
	gen = S.gen
	img_W = S.img_W
	border, between = S.border, S.between

	fig = show_nums(gen, figsize=(9, 9), W=img_W)
	plt.subplots_adjust(wspace=between, hspace=between,
	                    left=border, right=1 - border, bottom=border, top=1 - border)

	return fig,

def viz_latent(S, **unused):
	# if dis_q is None:

	assert 'int_q' in S

	if 'full_q' in S and S.full_q is not None:
		int_q = S.full_q
		if isinstance(S.full_q, distrib.Distribution):
			dis_int_q = S.full_q
		else:
			dis_int_q = None

	else:

		int_q = S.int_q
		dis_int_q = S.dis_int_q

	if dis_int_q is not None:
		int_q = int_q.loc
		dis_int_q = None

	print(int_q.shape)

	Xs = np.arange(int_q.shape[-1]) + 1
	inds = np.stack([Xs] * int_q.shape[0])

	vals = int_q.cpu().numpy()
	df1 = pd.DataFrame({'x': inds.reshape(-1), 'y': vals.reshape(-1)})

	if dis_int_q is not None:

		fig, ax = plt.subplots(figsize=(9, 3))

		df1['moment'] = 'mu'

		vals = dis_int_q.scale.log().cpu().numpy()
		df2 = pd.DataFrame({'x': inds.reshape(-1), 'y': vals.reshape(-1)})
		df2['moment'] = 'log(sigma)'

		df = pd.concat([df1, df2])

		hue = 'moment'
		split = False
		color = None
		palette = 'muted'
		inner = 'box'

		sns.violinplot(x='x', y='y', hue=hue,
		               data=df, split=split, color=color, palette=palette,
		               scale="count", inner=inner)
		plt.title('Distributions of Latent Dimensions')
		plt.xlabel('Dimension')
		plt.ylabel('Values')
		plt.legend(loc=8)
		plt.tight_layout()

	else:

		fig, ax = plt.subplots(figsize=(9, 3))

		df = df1

		hue = None
		split = False
		color = 'C0'
		inner = 'box'
		palette = None

		sns.violinplot(x='x', y='y', hue=hue,
		               data=df, split=split, color=color, palette=palette,
		               scale="count", inner=inner, gridsize=100, )
		plt.ylim(-3, 3)
		plt.title('Distributions of Latent Dimensions')
		plt.xlabel('Dimension')
		plt.ylabel('Values')
		plt.tight_layout()


	return fig,


def viz_interventions(S, **unused):

	A = S.A
	X = S.X
	q = S.q
	model = S.model
	img_W = S.img_W
	dataset = S.dataset

	int_q = S.int_q

	all_diffs = S.all_diffs


	# Intervention Effect

	vals = all_diffs.cpu().numpy()
	Xs = np.arange(vals.shape[-1]) + 1
	inds = np.stack([Xs] * vals.shape[0])
	df = pd.DataFrame({'x': inds.reshape(-1), 'y': vals.reshape(-1)})
	# df['moment']='log(sigma)'

	hue = None
	split = False
	color = 'C2'
	inner = 'box'
	palette = None

	fig, ax = plt.subplots(figsize=(9, 3))
	sns.violinplot(x='x', y='y', hue=hue,
	               data=df, split=split, color=color, palette=palette,
	               scale="count", inner=inner, gridsize=100)
	plt.title('Intervention Effect on Image')
	plt.xlabel('Dimension')

	plt.ylabel('Effect')
	plt.tight_layout()

	return fig,

def viz_traversals(S, **unused):

	walks = S.saved_walks

	anims = [util.Video(walk) for walk in walks]

	return anims


def _run_fid(generate, pbar=None):


	pass



def eval_prior_fid(S, pbar=None, **unused):

	model = S.model

	def generate(N):
		with torch.no_grad():
			return model.generate(N)

	return _run_fid(generate, pbar=pbar)

def eval_hybrid_fid(S, pbar=None, **unused):

	A = S.A

	Q = S.full_q
	assert Q is not None, 'no latent space'

	if isinstance(Q, distrib.Distribution):
		Q = Q.loc

	model = S.model

	def generate(N):

		idx = torch.randperm(len(Q))[:N]

		q = Q[idx].to(A.device)

		with torch.no_grad():
			gen = model.decode(util.shuffle_dim(q)).detach()

		return gen

	return _run_fid(generate, pbar=pbar)


def eval_disentanglement_metric(eval_fn, S, **unused):

	A = S.A
	model = S.model

	if model.enc is None:
		return None

	if 'repr_fn' not in S:
		S.repr_fn = dis_eval.representation_func(model, A.device)
	repr_fn = S.repr_fn

	if 'dis_dataset' not in S:
		S.dis_dataset = dis_eval.shapes3d.Shapes3D()
	dis_dataset = S.dis_dataset

	start = time.time()
	result = eval_fn(model='', representation_function=repr_fn, dataset=dis_dataset, seed=0)
	print('Took {:2.2f} s'.format(time.time()-start))

	return result

def make_dis_eval(eval_fn):
	def _eval_metric(S, **unused):
		return eval_disentanglement_metric(eval_fn, S, **unused)
	return _eval_metric


class Hybrid_Controller(train.Run_Manager):
	def __init__(self):
		super().__init__(load_fn=load_fn, run_model_fn=run_model,
		                 eval_fns=OrderedDict({

			                 # 'MIG': make_dis_eval(dis_eval.eval_mig),
			                 # 'DCI': make_dis_eval(dis_eval.eval_dci),
			                 'IRS': make_dis_eval(dis_eval.eval_irs),

			                 # 'SAP': make_dis_eval(dis_eval.eval_sap),
			                 # 'ModExp': make_dis_eval(dis_eval.eval_modularity_explicitness),
			                 # 'Unsup': make_dis_eval(dis_eval.eval_unsupervised),

							 # 'bVAE': make_dis_eval(dis_eval.eval_beta_vae),
			                 # 'FVAE': make_dis_eval(dis_eval.eval_factor_vae),
		                 }),
		                 viz_fns=OrderedDict({
							'original': viz_originals,
							'recs': viz_reconstructions,
							'gens': viz_generated,
							'hybrid': viz_hybrids,
							'latent': viz_latent,
							'effects': viz_interventions,
							'traversals': viz_traversals,
						}))


