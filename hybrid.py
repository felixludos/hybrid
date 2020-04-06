
import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as distrib

import h5py as hf
#%matplotlib tk
# plt.switch_backend('Agg')
# from sklearn.decomposition import PCA

import evaluate as dis_eval

import foundation as fd
from foundation import models
from foundation import data as datautils
from foundation.models import unsup
from foundation import util
from foundation import train as trn

MY_PATH = os.path.dirname(os.path.abspath(__file__))

trn.register_config_dir(os.path.join(MY_PATH, 'config'), recursive=True)

@fd.Component('wpp')
class Wasserstein_PP(fd.Generative, fd.Encodable, fd.Decodable, fd.Regularizable, fd.Visualizable, fd.Schedulable, fd.Trainable_Model):

	def __init__(self, A):

		encoder = A.pull('encoder')
		generator = A.pull('generator')
		discriminator = A.pull('discriminator')

		gan_wt = A.pull('gan_wt', 0.5)
		force_disc = A.pull('force_disc', False)
		disc_steps = A.pull('disc_steps', 1)
		enc_gan = A.pull('enc_gan', False)

		criterion_info = A.pull('criterion', None)
		latent_reg_wt = A.pull('latent_reg_wt', 1.)
		gan_reg_wt = A.pull('gan_reg_wt', 10.)
		gen_types = A.pull('gen_types', {'rec'})
		viz_force_gen = A.pull('viz_force_gen', False)

		rec_noise = A.pull('rec_noise', None)

		gan_warm_start = A.pull('gan_warm_start', None)

		if gan_wt == 1:
			gen_types = {'gen'}

		super().__init__(encoder.din, generator.din)
		self.step_counter = 0
		self.epoch_counter = 0

		self.enc = encoder
		self.gen = generator
		self.disc = discriminator

		self.latent_dim = self.gen.din

		self.reg_wts = util.NS()
		if latent_reg_wt > 0:
			self.reg_wts.latent = latent_reg_wt
		if gan_reg_wt > 0:
			self.reg_wts.gan = gan_reg_wt

		assert 0 <= gan_wt <= 1
		self.gan_wt = gan_wt
		if gan_wt == 0 and not force_disc:
			self.disc = None
		elif gan_wt == 1:
			self.enc = None

		gen_types = set(gen_types)
		assert gan_wt == 0 or (len(gen_types) and len(gen_types - {'rec', 'hybrid', 'gen'})==0), 'invalid: {}'.format(gen_types)
		self.gen_types = gen_types

		self.disc_steps = disc_steps if self.disc is not None else 0
		self.enc_gan = enc_gan and self.enc is not None

		self.criterion = util.get_loss_type(criterion_info) if isinstance(criterion_info, str) else util.get_loss_type(**criterion_info) # TODO: automate

		self.stats.new('reg-vae', 'reg-gan', 'wasserstein', 'reconstruction')
		self.viz_force_gen = viz_force_gen
		self._rec, self._real = None, None

		self.gan_warm_start = gan_warm_start
		assert self.gan_warm_start is None or 0 < self.gan_wt < 1, 'invalid gan wt: {}'.format(self.gan_wt)
		self.rec_noise = rec_noise

		self.set_optim()
		self.set_scheduler()

		self._dis_tracker = None #Disentanglement_Tracker(A, self, self.stats)

	def _img_size_limiter(self, imgs):
		H, W = imgs.shape[-2:]

		if H*W < 2e4: # upto around 128x128
			return imgs

		imgs = F.interpolate(imgs, size=(128,128))
		return imgs


	def _visualize(self, info, logger):
		if self._viz_counter % 2 == 0:
			if 'latent' in info and info.latent is not None:
				q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent

				shape = q.size()
				if len(shape) > 1 and np.product(shape) > 0:
					try:
						logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
						logger.add('histogram', 'latent-std', q.std(dim=0))
					except ValueError:
						print('\n\n\nWARNING: histogram just failed\n')
						print(q.shape, q.norm(p=2, dim=-1).shape)

			B, C, H, W = info.original.shape
			N = min(B, 8)

			if self._dis_tracker is not None:
				_dis_results = self._dis_tracker(self.stats)

			if 'reconstruction' in info:
				viz_x, viz_rec = info.original[:N], info.reconstruction[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', self._img_size_limiter(recs))
			elif self._rec is not None:
				viz_x, viz_rec = self._real[:N], self._rec[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', self._img_size_limiter(recs))

			if 'hygen' in info:
				logger.add('images', 'hygen', self._img_size_limiter(info.hygen[:N*2]))
			# elif 'fake' in info:
			# 	logger.add('images', 'hygen', info.fake[-N*2:])


			if 'fake' in info:
				logger.add('images', 'fake-img', self._img_size_limiter(info.fake[-N*2:]))

			if 'gen' not in info and self.viz_force_gen:
				with torch.no_grad():
					info.gen = self.generate(N*2)
			if 'gen' in info:
				viz_gen = info.gen[:2*N]
				logger.add('images', 'gen', self._img_size_limiter(viz_gen))

			logger.flush()

	def pre_epoch(self, mode, epoch):
		super().pre_epoch(mode, epoch)
		self.epoch_counter = epoch

	def _step(self, batch, out=None):
		self.step_counter += 1
		if out is None:
			out = util.TensorDict()

		x = batch[0]
		out.original = x
		B = x.size(0)

		rec = None
		if self.enc is not None:
			q = self.encode(x)
			out.latent = q

			qrec = q
			if self.rec_noise is not None:
				qrec += torch.randn_like(q)*self.rec_noise

			rec = self.decode(qrec)
			out.reconstruction = rec

		# train discriminator

		if self.disc is not None:

			if self.gan_wt == 1:
				fake = self.generate(B)
				out.gen = fake
			elif self.gan_wt > 0: # integrate latent space by hybridizing
				mix = self.hybridize(q)
				out.mix = mix
				fake = self.decode(mix)
			else: # using disc as feature match criterion
				fake = rec

			out.fake = fake

			real = x
			out.real = real

			# print('disc-verdicts', real.view(B,-1).sum(-1), fake.view(B,-1).sum(-1))

			verdict_real = self.judge(real)
			verdict_fake = self.judge(fake)

			out.vreal = verdict_real
			out.vfake = verdict_fake

			# print('disc-verdicts', verdict_real.shape, verdict_fake.shape)

			wasserstein = verdict_real.mean() - verdict_fake.mean()
			self.stats.update('wasserstein', wasserstein.detach())

			if self.gan_wt == 1:
				out.loss = wasserstein
			disc_loss = -wasserstein

			if 'gan' in self.reg_wts:
				gp_loss = unsup.grad_penalty(self.disc, real, fake)
				out.gp_loss = gp_loss
				self.stats.update('reg-gan', gp_loss.detach())
				disc_loss += self.reg_wts.gan * gp_loss

			if self.train_me():
				self.optim.disc.zero_grad()
				disc_loss.backward(retain_graph=True)
				self.optim.disc.step()

		# train encoder/generator

		if not self.train_me() or self.disc_steps <= 1 or self.step_counter % self.disc_steps == 0:

			if self.gan_wt > 0:

				verdict = 0.
				if 'rec' in self.gen_types:
					vrec = self.judge(rec)
					out.vrec = vrec
					verdict += vrec.mean()
				if 'hybrid' in self.gen_types:
					hygen = self.decode(self.hybridize(q))
					out.hygen = hygen
					vhyb = self.judge(hygen)
					out.vhyb = vhyb
					verdict += vhyb.mean()
				if 'gen' in self.gen_types:
					glat = self.sample_prior(B)
					gen = self.decode(glat)
					out.gen = gen
					vgen = self.judge(gen)
					out.vgen = vgen
					verdict += vgen.mean()

				verdict = verdict / len(self.gen_types)

				if self.train_me():

					self.optim.gen.zero_grad()
					if self.enc_gan:
						self.optim.enc.zero_grad()

					if self.gan_warm_start is None or self.epoch_counter > self.gan_warm_start:

						verdict.mul(-self.gan_wt).backward(retain_graph=True)

					if self.gan_wt == 1:
						self.optim.gen.step()
					elif not self.enc_gan:
						self.optim.enc.zero_grad()

			if self.gan_wt < 1:

				ae_loss = self.criterion(rec, x) / B
				out.rec_loss = ae_loss
				self.stats.update('reconstruction', ae_loss.detach())
				out.loss = ae_loss

				if 'latent' in self.reg_wts:

					reg_loss = self.regularize(q)
					out.reg_loss = reg_loss
					self.stats.update('reg-vae', reg_loss.detach())

					ae_loss += self.reg_wts.latent * reg_loss

				if self.train_me():
					if self.gan_wt == 0:
						self.optim.gen.zero_grad()
						self.optim.enc.zero_grad()

					ae_loss.mul(1-self.gan_wt).backward()

					self.optim.gen.step()
					self.optim.enc.step()


		return out


	def regularize(self, q, p=None):
		if p is None:
			p = self.sample_prior(q.size(0))
		return util.MMD(p, q)

	def hybridize(self, q):
		p = self.sample_prior(q.size(0))

		eta = torch.rand_like(q)
		mix = eta*p + (1-eta)*q

		return mix

	def encode(self, x):
		return self.enc(x)

	def decode(self, q):
		return self.gen(q)

	def judge(self, x):
		return self.disc(x)

	def forward(self, x):
		return self.encode(x)

	def sample_prior(self, N=1):
		return torch.randn(N, self.latent_dim, device=self.device)

	def generate(self, N=1):
		q = self.sample_prior(N)
		return self.decode(q)

@fd.Component('wgan')
class WGAN(Wasserstein_PP):
	def __init__(self, A):
		super().__init__(A)

		assert self.gan_wt == 1, 'gan_wt should be 1: {}'.format(self.gan_wt)

@fd.Component('vpp')
class WPP_VAE(Wasserstein_PP):
	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)

	def hybridize(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().hybridize(q)

	def regularize(self, q):
		return util.standard_kl(q).sum().div(q.loc.size(0))



class WPP_RAMMC(WPP_VAE):

	def __init__(self, A):

		n_samples = A.pull('rammc_samples')

		super().__init__(A)

		self.n_samples = n_samples

	def regularize(self, q):

		mu, sigma = q.loc, q.scale

		B = mu.size(0)

		# mu, sigma =



		raise NotImplementedError

@fd.Component('dwae')
class Dropin_WPP(Wasserstein_PP):
	def __init__(self, A):

		prob = A.pull('prob', 1)
		prob_max = A.pull('prob_max', None)

		super().__init__(A)

		self.shuffle = prob == 1
		
		probs = torch.ones(self.latent_dim) * prob \
			if prob_max is None or prob_max < prob \
			else torch.linspace(prob, prob_max, self.latent_dim)
		self.register_buffer('probs', probs.unsqueeze(0))

	def hybridize(self, q):

		hyb = util.shuffle_dim(q)
		
		if self.shuffle:
			return hyb
		
		sel = (torch.rand_like(q) - self.probs).gt(0).float()
		return q*sel + hyb*(1-sel)

@fd.Component('dout-wae')
class Dropout_WPP(Dropin_WPP):
	def hybridize(self, q):
		sel = (torch.rand_like(q) - self.probs).gt(0).float()
		return q * sel

@fd.Component('fwpp')
class Factor_WPP(Wasserstein_PP):
	def __init__(self, A):

		latent_disc = A.pull('latent_disc', None)
		reg_prior = A.pull('reg_prior', 0)
		ldisc_steps = A.pull('ldisc_steps', 1)
		ldisc_gp = A.pull('ldisc_gp', 10.)

		super().__init__(A)
		
		self.latent_disc = latent_disc
		# print('latent disc: {} {}'.format(latent_disc is not None, reg_prior))
		self.latent_disc_steps = ldisc_steps
		self.latent_disc_gp = ldisc_gp
		self.reg_step_counter = 0
		self.prior_wt = reg_prior
		if latent_disc is not None:
			self.stats.new('factor-ws', 'factor-gp')
		if reg_prior is not None and reg_prior > 0:
			assert 0 < reg_prior < 1, '{}'.format(reg_prior)
			self.stats.new('reg-prior', 'reg-factor')
	
	def regularize(self, q):

		qdis = q
		if isinstance(q, distrib.Distribution):
			q = q.loc
		
		mix = util.shuffle_dim(q)
		
		if self.latent_disc is None:
			reg = util.MMD(q, mix)
		else:
			self.reg_step_counter += 1
			
			vreal = self.latent_disc(mix)
			vfake = self.latent_disc(q)
			
			wasserstein = vreal.mean() - vfake.mean()
			self.stats.update('factor-ws', wasserstein.detach())
			
			loss = -wasserstein
			
			if self.latent_disc_gp is not None and self.latent_disc_gp > 0:
				lgp_loss = unsup.grad_penalty(self.latent_disc, mix, q)
				self.stats.update('factor-gp', lgp_loss.detach())
				loss += self.latent_disc_gp*lgp_loss
			
			if self.train_me():
				self.optim.latent_disc.zero_grad()
				loss.backward(retain_graph=True)
				self.optim.latent_disc.step()
				
			if self.latent_disc_steps <= 0 or self.reg_step_counter % self.latent_disc_steps == 0:
				reg = self.latent_disc(q).mean()
			else:
				reg = 0.
		
		if self.prior_wt is not None and self.prior_wt > 0:
			reg_prior = super().regularize(qdis)
			self.stats.update('reg-prior', reg_prior)
			self.stats.update('reg-factor', reg)
			reg = (1-self.prior_wt)*reg + self.prior_wt*reg_prior
			
		return reg

@fd.Component('swpp')
class Slice_WPP(Wasserstein_PP):

	def __init__(self, A):

		reg_prior = A.pull('reg_prior', 0)
		slices = A.pull('slices', '<>latent_dim')

		super().__init__(A)

		self.slices = slices
		self.reg_prior = reg_prior


	def sample_slices(self, N=None): # sampled D dim unit vectors
		if N is None:
			N = self.slices

		return torch.randn(N, self.latent_dim, device=self.device)

	def regularize(self, q, p=None):

		s = self.sample_slices()

		qd = F.cosine_similarity(q.unsqueeze(1), s.unsqueeze(0), dim=-1)
		qd = qd.sort(0)[0]

		if p is None:
			p = self.sample_prior(q.size(0))
		pd = F.cosine_similarity(p.unsqueeze(1), s.unsqueeze(0), dim=-1)
		pd = pd.sort(0)[0]

		return (qd - pd).abs().mean()


@fd.Component('fvpp')
class FactorVAE(Factor_WPP, WPP_VAE):
	pass

@fd.Component('dvae')
class DropinVAE(WPP_VAE, Dropin_WPP):
	pass

@fd.Component('fdvae')
class Dropin_FVAE(FactorVAE, Dropin_WPP):
	pass

@fd.Component('fdwae')
class Dropin_FWAE(Dropin_WPP, Factor_WPP):
	pass


class Filtered_Shapes3D(trn.datasets.Shapes3D):
	def __init__(self, dataroot, train=True, labels=False, dout=(3,64,64),
	             negative=False, override=False, replace=True):
		super().__init__(dataroot=dataroot, train=train, labels=True, dout=dout)

		self.labeled = labels

		if train or override:
			sel = self.selection(self.images, self.labels)

			lost = sel.sum().item()
			print('Filtering out {}/{} samples'.format(lost, len(self.images)))

			ridx = None
			if not negative and replace:
				try:
					ridx = self.replacements(self.images, self.labels)
				except NotImplementedError:
					sel = torch.logical_not(sel) # keep good samples
			elif negative:
				print('Negating selection')

			if ridx is None:
				self.images = self.images[sel]
				self.labels = self.labels[sel]
			else:

				swaps = len(ridx)

				print('Resampling from {} replacements'.format(swaps))

				copies, extra = lost // swaps, lost % swaps

				extra_idx = torch.randperm(swaps)[:extra]

				reps = torch.cat([ridx] * copies + [extra_idx])

				self.images[sel] = self.images[reps]
				self.labels[sel] = self.labels[reps]

		if not self.labeled:
			del self.labels

	# using self.images and self.labels
	def selection(self, images, labels): # return bools 1 if sample should be REMOVED
		raise NotImplementedError

	def replacements(self, images, labels): # return ints of possible replacements
		raise NotImplementedError

# ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']

@trn.Dataset('byfactor')
class ByFactor(trn.datasets.Shapes3D):
	# def __init__(self, dataroot, train=True, labels=False, dout=(3,64,64),
	#              factor='shape', vals=None, counts=None, seeds=None, det=True):

	def __init__(self, A):

		factor = A.pull('factor', 'shape')
		vals = A.pull('vals', None)

		counts = A.pull('counts', None)
		seeds = A.pull('seeds', None)

		det = A.pull('det', True)

		labeled = A.pull('labeled', False)
		A.labeled = True

		super().__init__(A)

		A.labeled = labeled
		self.labeled = labeled

		if not isinstance(factor, int):
			factor_name = factor
			factor_idx = self.factor_order.index(factor)
		else:
			factor_name = self.factor_order[factor]
			factor_idx = factor

		factor_size = self.factor_num_values[factor_name]

		if vals is None:
			vals = np.arange(factor_size)

		if counts is None:
			counts = [None]*factor_size
			print('WARNING: dataset is not filtering out any samples')
		else:
			counts = [(cnt if cnt is None or cnt >= 0 else None) for cnt in counts]


		if seeds is None:
			seeds = np.arange(factor_size) if det else [None]*factor_size

		assert len(vals) == factor_size, 'invalid len: {} vs {}'.format(factor_size, len(vals))
		assert len(counts) == factor_size, 'invalid len: {} vs {}'.format(factor_size, len(counts))
		assert len(seeds) == factor_size, 'invalid len: {} vs {}'.format(factor_size, len(seeds))

		self.factor_name = factor_name
		self.factor_idx = factor_idx
		self.factor_size = factor_size

		sel = self._subsample(vals, counts, seeds)

		print('Filtering out {}/{} samples'.format(len(self.images)-len(sel), len(self.images)))

		self.images = self.images[sel]
		self.labels = self.labels[sel]

		if True or not self.labeled: # testing
			del self.labels
			self.dout = self.din
			self.labeled = False

	def _filter(self, samples, val, num=None, seed=None):

		sel = torch.isclose(samples, torch.tensor(val).float())

		indices = torch.arange(len(samples))[sel]

		if num is not None and num < len(indices):
			num = min(len(indices), num)
			indices = util.subset(indices, num, seed)

		return indices

	def _subsample(self, vals, counts, seeds):

		samples = self.labels[:,self.factor_idx]

		inds = []
		for val, cnt, seed in zip(vals, counts, seeds):
			inds.append(self._filter(samples, val, num=cnt, seed=seed))

		inds = np.concatenate(inds)

		return inds

@trn.Dataset('atari')
class Atari_Playback(datautils.Testable_Dataset, datautils.Info_Dataset):
	def __init__(self, A, game=None):

		dataroot = A.pull('dataroot')

		if game is None:
			game = A.pull('game') # available games: Asterix, Seaquest, SpaceInvaders, MsPacman

		assert game in {'Asterix', 'Seaquest', 'SpaceInvaders', 'MsPacman'}, 'game not available'

		train = A.pull('train', True)

		dataroot = os.path.join(dataroot, 'atari', game)

		total = [n for n in sorted(os.listdir(dataroot)) if '.h5' in n]

		# assert len(total) == 16 # TODO: maybe change to 20 (requires recollecting data)

		fnames = total[:-4] if train else total[-4:]

		print('Found {} datafiles. Using {}'.format(len(total), len(fnames)))

		raw = []

		for fname in fnames:

			with hf.File(os.path.join(dataroot, fname), 'r') as f:
				raw.append(f['images'][()].reshape(-1))

		raw = np.asarray(raw).reshape(-1)

		self.raw = raw

		dummy = util.str_to_rgb(raw[0])

		H, W, C = dummy.shape

		din = 3, 128, 128

		super().__init__(din=din, dout=din)


	def __len__(self):
		return len(self.raw)

	def __getitem__(self, idx):
		img = torch.from_numpy(util.str_to_rgb(self.raw[idx])).permute(2,0,1).float().unsqueeze(0) / 255.
		img = F.interpolate(img, size=(128,128), mode='bilinear').squeeze(0)
		return img,


@trn.Dataset('transfer')
class Transfer_Dataset(datautils.Info_Dataset):
	def __init__(self, A):
		'''
		Train a model that was pretrained on 'old' dataset to generalize to 'new'
		'new' should be much smaller than 'old'
		'''

		# assert False, 'pre_epoch not setup for datasets yet'

		# load = A.pull('load') # Testing
		# print('Using pretrained model: {}'.format(load))

		assert 'old' in A and 'new' in A, 'no datasets to compare'

		budget = A.pull('budget', None)
		old2new_ratio = A.pull('old2new_ratio', 1)

		if budget is not None:
			if A.old._type == 'dataset/byfactor' and 'counts' in A.old:
				cnts = A.old.counts
				A.old.counts = [(budget if c == -2 else c) for c in cnts]
				print('Replaced counts in old: from {} to {}'.format(cnts, A.old.counts))
			if A.new._type == 'dataset/byfactor' and 'counts' in A.new:
				cnts = A.new.counts
				A.new.counts = [(budget if c == -2 else c) for c in cnts]
				print('Replaced counts in new: from {} to {}'.format(cnts, A.new.counts))
				budget = None

		old = A.pull('old')

		try:
			new = self._derive_new_from_old(old)
		except NotImplementedError:
			new = None

		if new is None:
			new = A.pull('new')

		assert old.din == new.din and old.dout == new.dout, 'datasets are not compatible'

		super().__init__(old.din, old.dout)


		self.new = new
		self.old = old
		self._limit_new(budget)

		self.num_old = min(int(old2new_ratio * len(self.new)), len(self.old))
		self.num_new = len(self.new)

		self.resample_old()

	def _limit_new(self, budget):
		return
		if budget is not None:
			new = self.new
			if len(new) < budget:
				raise Exception('Not enough samples in new dataset {} for a budget of {}'.format(len(new), budget))
			elif len(new) > budget:
				if isinstance(new, ByFactor):
					cnts = new.counts

				inds = torch.randperm(len(new))[:budget]
				new = datautils.Subset_Dataset(new, inds)
				raise NotImplementedError

			self.new = new

	def _derive_new_from_old(self, old):
		raise NotImplementedError

	def __len__(self):
		return self.num_old + self.num_new

	def resample_old(self):
		self.old_inds = torch.randint(0, len(self.old), size=(self.num_old,))

	def __getitem__(self, item):
		if item < self.num_new:
			return self.new[item]
		return self.old[self.old_inds[item-self.num_new]]

	def pre_epoch(self, mode, epoch):
		if mode == 'train':
			# print('Replacing old samples')# Testing
			self.resample_old()
		self.old.pre_epoch(mode, epoch)
		self.new.pre_epoch(mode, epoch)

	def post_epoch(self, mode, epoch, stats=None):
		self.old.post_epoch(mode, epoch, stats=stats)
		self.new.post_epoch(mode, epoch, stats=stats)
# trainutils.register_dataset('transfer', Transfer_Dataset)










# Deep Hybridization - using AdaIN

@fd.Component('adain')
class AdaIN(fd.Model):
	def __init__(self, A):

		qdim = A.pull('ada_noise', '<>latent_dim')
		cdim = A.pull('features', '<>din')

		pixelwise = A.pull('pixelwise', False)

		ndim = cdim[0] if isinstance(cdim, (tuple, list)) and not pixelwise else cdim

		if 'net' in A:
			A.net.din = qdim
			A.net.dout = ndim

		net = A.pull('net', None)

		if isinstance(qdim, (tuple,list)) and isinstance(cdim, (tuple,list)):
			raise NotImplementedError

		super().__init__(cdim, cdim)

		self.net = net
		self.noise = None

	def default_noise(self, n):
		return torch.zeros(n.size(0), *self.dout, device=n.device, dtype=n.dtype)

	def process_noise(self, n):
		if self.net is None:
			return self.default_noise(n)
		return self.net(n)

	def include_noise(self, x, q):

		if len(x.shape) != len(q.shape):
			assert len(x.shape) > len(q.shape), 'not the right sizes: {} vs {}'.format(x.shape, q.shape)
			q = q.view(*q.shape, *(1,)*(len(x.shape)-len(q.shape)))

		return x + q

	def set_noise(self, n):
		self.noise = n

	def forward(self, x, n=None):
		if n is None and self.noise is not None:
			n = self.noise
			self.noise = None
		if n is not None:
			q = self.process_noise(n)
			x = self.include_noise(x, q)
		return x
# trainutils.register_model('ada-in', AdaIN)

@fd.Component('norm-ada-in')
class Norm_AdaIN(AdaIN):
	def __init__(self, A):
		if 'net' in A:
			assert '_mod' in A.net and A.net._mod == 'normal', 'must output a distribution'
		super().__init__(A)

	def include_noise(self, x, q):
		mu, sigma = q.loc, q.scale
		if len(x.shape) != len(mu.shape):
			assert len(x.shape) > len(mu.shape), 'not the right sizes: {} vs {}'.format(x.shape, mu.shape)
			mu = mu.view(*mu.shape, *(1,)*(len(x.shape)-len(mu.shape)))
			sigma = sigma.view(*sigma.shape, *(1,) * (len(x.shape) - len(sigma.shape)))

		return sigma*x + mu
# trainutils.register_model('norm-ada-in', Norm_AdaIN)

@fd.Component('adain-double-dec')
class AdaIn_Double_Decoder(models.Double_Decoder):

	def __init__(self, A):

		adain_latent_dim = A.pull('adain_latent_dim', 0)
		full_latent_dim = A.pull('latent_dim', '<>din')

		const_start = False
		init_latent_dim = full_latent_dim - adain_latent_dim
		assert init_latent_dim >= 0, 'invalid: {}'.format(A.latent_dim)
		if init_latent_dim == 0:
			init_latent_dim = 1
			const_start = True
		A.latent_dim = init_latent_dim
		A.full_latent_dim = full_latent_dim
		A.adain_latent_dim = adain_latent_dim

		super().__init__(A)

		A.latent_dim = full_latent_dim

		if adain_latent_dim is not None:
			self.din = full_latent_dim

		self.adain_latent_dim = adain_latent_dim

		self.init_latent_dim = init_latent_dim
		self.const_start = const_start

	def _create_layers(self, chns, factors, internal_channels, squeeze, A):

		between_blocks = len(chns)-2
		adains = A.pull('adains', [True] * between_blocks)
		try:
			len(adains)
		except TypeError:
			adains = [adains]
		if len(adains) != between_blocks:
			adains = adains * between_blocks
		adains = iter(adains)

		splits = A.pull('splits', None)
		if splits is not None:
			try:
				len(splits)
			except TypeError:
				splits = [splits]
			if len(splits) != between_blocks:
				splits = splits * between_blocks
			splits = splits[:between_blocks]

		self.splits = splits
		if splits is not None:
			splits = iter(splits)
		full_latent = A.pull('full_latent_dim')

		nonlin = A.pull('nonlin', 'elu')
		output_nonlin = A.pull('output_nonlin', None)
		output_norm_type = A.pull('output_norm_type', None)

		up_type = A.pull('up_type', 'bilinear')
		norm_type = A.pull('norm_type', None)
		residual = A.pull('residual', False)

		last_chn = chns[-2:]
		chns = chns[:-1]

		layers = []

		alayers = []

		for ichn, ochn in zip(chns, chns[1:]):
			layers.append(
				models.DoubleDeconvLayer(in_channels=ichn, out_channels=ochn, factor=next(factors),
				                            up_type=up_type, norm=norm_type,
				                            nonlin=nonlin, output_nonlin=nonlin,
				                            internal_channels=next(internal_channels), squeeze=next(squeeze),
				                            residual=residual,
				                            )
			)
			if next(adains):

				dim = next(splits) if splits is not None else full_latent
				if dim is not None:
					A.adain.ada_noise = dim
					A.adain.features = ochn

				adain = A.pull('adain', ref=False)

				if 'ada_noise' in A.adain:
					del A.adain.ada_noise
				if 'features' in A.adain:
					del A.adain.features

				alayers.append(adain)
				layers.append(adain)
		layers.append(
			models.DoubleDeconvLayer(in_channels=last_chn[0], out_channels=last_chn[1], factor=next(factors),
			                            up_type=up_type, norm=output_norm_type,
			                            nonlin=nonlin, output_nonlin=output_nonlin,
			                            internal_channels=next(internal_channels), squeeze=next(squeeze),
			                            residual=residual,
			                            )
		)

		self.ada_ins = alayers

		return nn.ModuleList(layers)

	def forward(self, q):

		if self.const_start:
			init = torch.ones(q.size(0), 1, dtype=q.dtype, device=q.device)
		else:
			init, q = q[...,:self.init_latent_dim], q[...,self.init_latent_dim:]

		noises = [q]*len(self.ada_ins)
		if self.splits is not None:
			noises = torch.split(q,self.splits, dim=-1)

		for adain, noise in zip(self.ada_ins, noises):
			adain.set_noise(noise)

		return super().forward(init)

# trainutils.register_model('adain-double-dec', AdaIn_Double_Decoder)


class Disentanglement_Tracker(object): # Interventional Robustness

	def __init__(self, A, model, stats):
		# print('\n\n\n\n\n\n\nWARNING: not using tracker\n\n\n\n')

		self.repr_fn = None

		if A.dataset.name == '3dshapes' and model.enc is not None:
			print('Will track disentanglement score throughout training')
			self.repr_fn = dis_eval.representation_func(model, A.device)

			self.dis_dataset = dis_eval.shapes3d.Shapes3D()

			self.register(stats)

		pass

	def register(self, stats):
		stats.new('dtngle-irs', 'dtngle-dim')

	def __call__(self, stats):
		if self.repr_fn is not None:

			result = dis_eval.eval_irs(model='', representation_function=self.repr_fn, dataset=self.dis_dataset, seed=0)

			stats.update('dtngle-irs', float(result['IRS']))
			stats.update('dtngle-dim', float(result['num_active_dims']))

			return result


### Required


def get_data(A, mode='train'):
	return trn.default_load_data(A, mode=mode)

def get_model(A):
	return trn.default_create_model(A)

def get_name(A):
	assert 'name' in A, 'Must provide a name manually'
	return A.name

def main(argv=None):
	return trn.main(argv=argv, get_data=get_data, get_model=get_model, get_name=get_name)

if __name__ == '__main__':
	sys.exit(main(sys.argv))




# class Asterix_Playback(Atari_Playback):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, game='Asterix', **kwargs)
# trainutils.register_dataset('asterix', Asterix_Playback)
#
# class Seaquest_Playback(Atari_Playback):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, game='Seaquest', **kwargs)
# trainutils.register_dataset('seaquest', Seaquest_Playback)
#
# class SpaceInvaders_Playback(Atari_Playback):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, game='SpaceInvaders', **kwargs)
# trainutils.register_dataset('spaceinv', SpaceInvaders_Playback)
#
# class Pacman_Playback(Atari_Playback):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, game='MsPacman', **kwargs)
# trainutils.register_dataset('pacman', Pacman_Playback)

# @fd.AutoModifier('transfer')
# class Transfer_Dataset(datautils.Info_Dataset):
# 	def __init__(self, A, new=None, budget=None, old2new_ratio=None):
# 		'''
# 		Train a model that was pretrained on 'old' dataset to generalize to 'new'
# 		'new' should be much smaller than 'old'
# 		'''
#
# 		# assert False, 'pre_epoch not setup for datasets yet'
#
# 		load = A.pull('load')
# 		print('Using pretrained model: {}'.format(load))
#
# 		if new is None:
# 			new = A.pull('_new')
#
# 		if budget is None:
# 			budget = A.pull('budget', None)
# 		if old2new_ratio is None:
# 			old2new_ratio = A.pull('old2new_ratio', 1)
#
# 		assert self.din == new.din and self.dout == new.dout, 'datasets are not compatible'
#
# 		super().__init__(A)
#
# 		if budget is not None:
# 			inds = torch.randperm(len(new))[:budget]
# 			new = datautils.Subset_Dataset(new, inds)
# 		self.new = new
#
# 		self.num_old = min(int(old2new_ratio * len(self.new)), len(self))
# 		self.num_new = len(self.new)
#
# 		self.resample_old()
#
# 	def __len__(self):
# 		return self.num_old + self.num_new
#
# 	def resample_old(self):
# 		self.old_inds = torch.randint(0, len(self.old), size=(self.num_old,))
#
# 	def __getitem__(self, item):
# 		if item < self.num_new:
# 			return self.new[item]
# 		return super().__getitem__(self.old_inds[item-self.num_new])
#
# 	def pre_epoch(self, mode, epoch):
# 		if mode == 'train':
# 			print('Replacing old samples')
# 			self.resample_old()
# trainutils.register_dataset('transfer', Transfer_Dataset)



# trainutils.register_dataset('byfactor', ByFactor)

# class RBall_Shapes3D(Filtered_Shapes3D):
#
# 	def selection(self, images, labels): # all non-red balls
# 		return torch.logical_not(labels[:, 2].isclose(torch.tensor(0.))) * labels[:, -2].isclose(torch.tensor(2.))
#
# 	def replacements(self, images, labels): # any red balls
# 		return torch.arange(len(images))[labels[:, 2].isclose(torch.tensor(0.)) * labels[:, -2].isclose(torch.tensor(2.))]
# trainutils.register_dataset('redball-3dshapes', RBall_Shapes3D)
#
# class RBBall_Shapes3D(Filtered_Shapes3D):
#
# 	def selection(self, images, labels): # all non-RGB balls
# 		return torch.logical_not(labels[:, 2].isclose(torch.tensor(0.))
#                   + labels[:, 2].isclose(torch.tensor(0.7))) * labels[:, -2].isclose(torch.tensor(2.))
#
# 	def replacements(self, images, labels): # any RGB balls
# 		return torch.arange(len(images))[(labels[:, 2].isclose(torch.tensor(0.))
#                     + labels[:, 2].isclose(torch.tensor(0.7))) * labels[:, -2].isclose(torch.tensor(2.))]
# trainutils.register_dataset('rbball-3dshapes', RBBall_Shapes3D)
#
# class RGBBall_Shapes3D(Filtered_Shapes3D):
#
# 	def selection(self, images, labels): # all non-RGB balls
# 		return torch.logical_not(labels[:, 2].isclose(torch.tensor(0.))
#                   + labels[:, 2].isclose(torch.tensor(0.3))
#                   + labels[:, 2].isclose(torch.tensor(0.7))) * labels[:, -2].isclose(torch.tensor(2.))
#
# 	def replacements(self, images, labels): # any RGB balls
# 		return torch.arange(len(images))[(labels[:, 2].isclose(torch.tensor(0.))
#                     + labels[:, 2].isclose(torch.tensor(0.3))
#                     + labels[:, 2].isclose(torch.tensor(0.7))) * labels[:, -2].isclose(torch.tensor(2.))]
# trainutils.register_dataset('rgbball-3dshapes', RGBBall_Shapes3D)

#
# class NoCap_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all capsule
# 		return labels[:, -2].isclose(torch.tensor(3.))
# trainutils.register_dataset('nocap-3dshapes', NoCap_Shapes3D)

# class Cylinder_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all non-cylinder
# 		return torch.logical_not(labels[:, -2].isclose(torch.tensor(1.)))
# trainutils.register_dataset('cylinder-3dshapes', Cylinder_Shapes3D)
#
# class Ball_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all non-ball
# 		return torch.logical_not(labels[:, -2].isclose(torch.tensor(2.)))
# trainutils.register_dataset('ball-3dshapes', Ball_Shapes3D)
#
# class Cube_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all non-ball
# 		return torch.logical_not(labels[:, -2].isclose(torch.tensor(2.)))
# trainutils.register_dataset('ball-3dshapes', Ball_Shapes3D)
#
# class CylBall_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all non-(ball or cyl)
# 		return torch.logical_not(labels[:, -2].isclose(torch.tensor(1.)) + labels[:, -2].isclose(torch.tensor(2.)))
# trainutils.register_dataset('cylball-3dshapes', CylBall_Shapes3D)


# class Cylinder_Shapes3D(Filtered_Shapes3D):
# 	def selection(self, images, labels): # all cylinder
# 		return labels[:, -2].isclose(torch.tensor(1.))
# trainutils.register_dataset('cylinder-3dshapes', Cylinder_Shapes3D)

