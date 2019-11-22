
import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.distributions as distrib

#%matplotlib tk
# plt.switch_backend('Agg')
# from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation import data as datautils
from foundation.models import unsup
from foundation import util
from foundation import train

MY_PATH = os.path.dirname(os.path.abspath(__file__))

train.register_config('dspr', os.path.join(MY_PATH, 'config', 'dspr.yaml'))
train.register_config('hybrid', os.path.join(MY_PATH, 'config', 'basics.yaml'))
train.register_config('pycharm', os.path.join(MY_PATH, 'config', 'pycharm.yaml'))

class Wasserstein_PP(fd.Generative, fd.Encodable, fd.Decodable, fd.Regularizable, fd.Visualizable, fd.Trainable_Model):

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


		super().__init__(encoder.din, generator.din)
		self.step_counter = 0

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

		assert gan_wt == 0 or (len(gen_types) and len(gen_types - {'rec', 'hybrid', 'gen'})==0), 'invalid: {}'.format(gen_types)
		self.gen_types = gen_types

		self.disc_steps = disc_steps if self.disc is not None else 0
		self.enc_gan = enc_gan and self.enc is not None

		self.criterion = util.get_loss_type(criterion_info) if isinstance(criterion_info, str) else util.get_loss_type(**criterion_info) # TODO: automate

		self.stats.new('reg-vae', 'reg-gan', 'wasserstein', 'reconstruction')
		self.viz_force_gen = viz_force_gen
		self._rec, self._real = None, None

		self.set_optim()

	def _visualize(self, info, logger):
		if self._viz_counter % 5 == 0:
			if 'latent' in info and info.latent is not None:
				q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent

				logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
				logger.add('histogram', 'latent-std', q.std(dim=0))

			B, C, H, W = info.original.shape
			N = min(B, 8)

			if 'reconstruction' in info:
				viz_x, viz_rec = info.original[:N], info.reconstruction[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', recs)
			elif self._rec is not None:
				viz_x, viz_rec = self._real[:N], self._rec[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', recs)

			if 'hygen' in info:
				logger.add('images', 'hygen', info.hygen[:N*2])
			# elif 'fake' in info:
			# 	logger.add('images', 'hygen', info.fake[-N*2:])


			if 'fake' in info:
				logger.add('images', 'fake-img', info.fake[-N*2:])

			if 'gen' not in info and self.viz_force_gen:
				with torch.no_grad():
					info.gen = self.generate(N*2)
			elif 'gen' in info:
				viz_gen = info.gen[:2*N]
				logger.add('images', 'gen', viz_gen)

			logger.flush()


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

			rec = self.decode(q)
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

		if self.disc_steps <= 1 or self.step_counter % self.disc_steps == 0:

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


	def regularize(self, q):
		return util.MMD(self.sample_prior(q.size(0)), q)

	def hybridize(self, q):
		p = self.sample_prior(q.size(0))

		if q is None:
			return p

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

train.register_model('wpp', Wasserstein_PP)


class WPP_VAE(Wasserstein_PP):
	def __init__(self, A):
		min_log_std = A.pull('min_log_std', -3)

		raise NotImplementedError('Gaussian_Encoder is missing')

		super().__init__(A)

		self.min_log_std = min_log_std

	def decode(self, q=None, N=None):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q, N)

	def encode(self, x):
		if self.enc is None:
			return None

		q = self.enc(x)

		mu = q.narrow(-1, 0, self.latent_dim)
		logsigma = q.narrow(-1, self.latent_dim, self.latent_dim)
		if self.min_log_std is not None:
			logsigma = logsigma.clamp(min=self.min_log_std)
		sigma = logsigma.exp()

		return distrib.Normal(loc=mu, scale=sigma)

	def hybridize(self, q=None, N=None):
		if q is None:
			return self.sample_prior(N)
		return q

	def regularize(self, q):
		return util.standard_kl(q).sum().div(q.loc.size(0))

train.register_model('wpp-vae', WPP_VAE)

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

train.register_model('dropin', Dropin_WPP)

class Dropout_WPP(Dropin_WPP):

	def hybridize(self, q):
		sel = (torch.rand_like(q) - self.probs).gt(0).float()
		return q * sel

train.register_model('dropout', Dropout_WPP)

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
			reg_prior = super().regularize(q)
			self.stats.update('reg-prior', reg_prior)
			self.stats.update('reg-factor', reg)
			reg = (1-self.prior_wt)*reg + self.prior_wt*reg_prior
			
		return reg

train.register_model('factor', Factor_WPP)

class Dropin_FWAE(Dropin_WPP, Factor_WPP):
	pass

train.register_model('factor-dropin', Dropin_FWAE)


def get_data(A, mode='train'):
	return train.default_load_data(A, mode=mode)

def get_model(A):
	return train.default_create_model(A)

def get_name(A):
	assert 'name' in A, 'Must provide a name manually'
	return A.name


if __name__ == '__main__':
	sys.exit(train.main(get_data=get_data, get_model=get_model, get_name=get_name))



#
# def get_options():
# 	parser = train.get_parser()
#
# 	parser.add_argument('--gan-wt', type=float, default=.5)
#
# 	parser.add_argument('--disc-steps', type=int, default=1)
# 	parser.add_argument('--disc-lr-factor', type=float, default=1.)
# 	parser.add_argument('--disc-gp', type=float, default=10.)
#
# 	parser.add_argument('--fake-gen', action='store_true')
# 	parser.add_argument('--fake-hyb', action='store_true')
# 	parser.add_argument('--no-fake-rec', dest='fake_rec', action='store_false')
# 	parser.add_argument('--enc-gan', action='store_true')
#
# 	parser.add_argument('--prior-wt', type=float, default=0.)
# 	parser.add_argument('--latent-disc-fc', type=int, default=None, nargs='+')
#
# 	parser.add_argument('--beta', type=float, default=None)
# 	parser.add_argument('--criterion', type=str, default='bce')
#
# 	parser.add_argument('--disc-fc', type=int, nargs='+', default=None)
# 	parser.add_argument('--enc-fc', type=int, nargs='+', default=None)
#
# 	parser.add_argument('--disc-lr', type=float, default=None)
# 	parser.add_argument('--enc-lr', type=float, default=None)
#
# 	parser.add_argument('--prob', type=float, default=1)
# 	parser.add_argument('--prob-max', type=float, default=0)
#
# 	parser.add_argument('--disc-optim-type', type=str, default=None)
# 	parser.add_argument('--enc-optim-type', type=str, default=None)
#
# 	# old
#
# 	parser.add_argument('--vae-weight', type=float, default=None)
# 	parser.add_argument('--feature-match', action='store_true')
# 	parser.add_argument('--noisy-rec', action='store_true')
# 	parser.add_argument('--noisy-gan', action='store_true')
# 	parser.add_argument('--pred-std', action='store_true')
#
# 	parser.add_argument('--viz-force-gen', action='store_true')
#
# 	parser.add_argument('--vae-scale', type=float, default=1)
# 	parser.add_argument('--gan-scale', type=float, default=1)
#
# 	parser.add_argument('--splits', type=int, default=2)
#
# 	return parser
#
#
# def get_data(args):
# 	if not hasattr(args, 'dataroot'):
# 		args.dataroot = '/is/ei/fleeb/workspace/local_data/'
#
# 	dataroot = args.dataroot
#
# 	if args.dataset == 'dsprites':
#
# 		def fmt(batch):
# 			batch['imgs'] = torch.from_numpy(batch['imgs']).unsqueeze(0)
# 			return batch['imgs'].float(), batch['latents_values']
#
# 		path = os.path.join(dataroot, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
# 		dataset = datautils.Npy_Loader_Dataset(path, keys=['imgs', 'latents_values'])
#
# 		dataset = datautils.Format_Dataset(dataset, fmt)
#
# 		args.din = 1, 64, 64
#
# 		return dataset,
#
# 	elif args.dataset == '3dshapes':
#
# 		def fmt(batch):
# 			batch['images'] = torch.from_numpy(batch['images']).permute(2, 0, 1)
# 			return batch['images'].float().div(255), batch['labels']
#
# 		path = os.path.join(dataroot, '3dshapes.h5')
# 		dataset = datautils.H5_Dataset(path, keys=['images', 'labels'])
#
# 		dataset = datautils.Format_Dataset(dataset, fmt)
#
# 		args.din = 3, 64, 64
#
# 		return dataset,
#
# 	return foundation.old.load_data(args=args)
#
#
def __old_get_model(args):
	if not hasattr(args, 'feature_match'):  # bwd compatibility
		args.feature_match = False

	if not hasattr(args, 'noisy_gan'):
		args.noisy_gan = False

	model_args = {}

	if hasattr(args, 'noisy_rec') and args.noisy_rec:
		model_args['noise_std'] = args.beta

	if args.model_type in {'wpp-vae'}:
		args.pred_std = True

	enc_out = args.latent_dim
	if args.pred_std:
		enc_out *= 2

	if args.disc_fc is None:
		args.disc_fc = args.fc.copy()[::-1]
	if args.enc_fc is None:
		args.enc_fc = args.fc.copy()[::-1]

	encoder = models.Conv_Encoder(args.din, latent_dim=enc_out,

	                              nonlin=args.nonlin, output_nonlin=None,

	                              channels=args.channels[::-1], kernels=args.kernels[::-1],
	                              strides=args.strides, factors=args.factors[::-1],
	                              down=args.downsampling, norm_type=args.norm_type,

	                              hidden_fc=args.enc_fc)

	generator = models.Conv_Decoder(args.din, latent_dim=args.latent_dim,

	                                nonlin=args.nonlin, output_nonlin='sigmoid',

	                                channels=args.channels, kernels=args.kernels,
	                                ups=args.factors,
	                                upsampling=args.upsampling, norm_type=args.norm_type,

	                                hidden_fc=args.fc)

	discriminator = models.Conv_Encoder(args.din, latent_dim=1,

	                                    nonlin=args.nonlin, output_nonlin=None,

	                                    channels=args.channels[::-1], kernels=args.kernels[::-1],
	                                    strides=args.strides, factors=args.factors[::-1],
	                                    down=args.downsampling, norm_type=args.norm_type,

	                                    hidden_fc=args.disc_fc)

	latent_disc = None
	# if 'wpp' in args.model_type:
	kwargs = {}

	model_cls = Wasserstein_PP
	if 'vae' in args.model_type:
		model_cls = WPP_VAE
	# kwargs['min_log_std'] = args.min_log_std

	if 'fwae' in args.model_type:
		model_cls = Factor_WAE

		if args.latent_disc_fc is not None:
			latent_disc = models.make_MLP(args.latent_dim, 1, hidden_dims=args.latent_disc_fc,
			                              nonlin=args.nonlin)
			print('Using discriminator in the latent space:')
			print(latent_disc)

		kwargs.update({
			'latent_disc': latent_disc,
			'reg_prior': args.prior_wt,
		})
		print('Prior wt: {}'.format(args.prior_wt))

	if 'dropin' in args.model_type:
		model_cls = Dropin_WPP
		kwargs['prob'] = args.prob
		kwargs['prob_max'] = args.prob_max
		print('Prob resampling: {}'.format(args.prob))

	if args.model_type == 'fwae-dropin':
		model_cls = Dropin_FWAE

	# encoder.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	# generator.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	# discriminator.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

	criterion = util.get_loss_type(args.criterion, reduction='sum')
	if args.feature_match and args.gan_wt < 1:
		# layers = [conv.conv for conv in discriminator.conv] \
		#          + [fc for fc in discriminator.fc if isinstance(fc, nn.Linear)][:-1]

		layers = discriminator.conv

		criterion = models.Feature_Match(layers,
		                                 criterion=util.get_loss_type(args.criterion, reduction='sum'),
		                                 model=discriminator)

		print('Using feature match ({} layers)'.format(len(layers)))
	print(criterion)

	gen_types = {'rec'}
	if not args.fake_rec:
		gen_types.remove('rec')
	if args.fake_gen:
		gen_types.add('gen')
	if args.fake_hyb:
		gen_types.add('hybrid')

	if args.gan_wt == 1:
		print('Only GAN')
		args.viz_criterion_args = None
		gen_types = {'gen'}
	if args.gan_wt == 0:
		print('Only VAE')
	print('Using gan-wt: {}'.format(args.gan_wt))
	print('Using gen_types: {}'.format(gen_types))

	model = model_cls(encoder, generator, discriminator,
	                  gan_wt=args.gan_wt, force_disc=args.feature_match, disc_steps=args.disc_steps,
	                  enc_gan=args.enc_gan,
	                  criterion=criterion, latent_reg_wt=args.beta, gan_reg_wt=args.disc_gp,
	                  gen_types=gen_types, viz_force_gen=args.viz_force_gen, **kwargs
	                  )

	if args.disc_lr is None:
		args.disc_lr = args.lr * args.disc_lr_factor
	if args.enc_lr is None:
		args.enc_lr = args.lr
	if not hasattr(args, 'disc_optim_type') or args.disc_optim_type is None:
		args.disc_optim_type = args.optim_type
	if not hasattr(args, 'enc_optim_type') or args.enc_optim_type is None:
		args.enc_optim_type = args.optim_type

	model.set_optim(util.Complex_Optimizer(
		gen=util.get_optimizer(args.optim_type, generator.parameters(), lr=args.lr, weight_decay=args.weight_decay,
		                       momentum=args.momentum, beta1=args.beta1, beta2=args.beta2),
	))

	if model.enc is not None:
		model.optim.enc = util.get_optimizer(args.enc_optim_type, encoder.parameters(), lr=args.enc_lr,
		                                     weight_decay=args.weight_decay,
		                                     momentum=args.momentum, beta1=args.beta1, beta2=args.beta2)

	if model.disc is not None:
		model.optim.disc = util.get_optimizer(args.disc_optim_type, discriminator.parameters(), lr=args.disc_lr,
		                                      weight_decay=args.weight_decay,
		                                      momentum=args.momentum, beta1=args.beta1, beta2=args.beta2)

	if latent_disc is not None:
		model.optim.latent_disc = util.get_optimizer(args.optim_type, latent_disc.parameters(), lr=args.lr,
		                                             weight_decay=args.weight_decay,
		                                             momentum=args.momentum, beta1=args.beta1, beta2=args.beta2)

	return model



