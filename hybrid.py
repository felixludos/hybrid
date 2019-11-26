
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

train.register_config('factor', os.path.join(MY_PATH, 'config', 'factor.yaml'))
train.register_config('dropin', os.path.join(MY_PATH, 'config', 'dropin.yaml'))
train.register_config('dropin_factor', os.path.join(MY_PATH, 'config', 'dropin_factor.yaml'))

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

		rec_noise = A.pull('rec_noise', None)

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

		gen_types = set(gen_types)
		assert gan_wt == 0 or (len(gen_types) and len(gen_types - {'rec', 'hybrid', 'gen'})==0), 'invalid: {}'.format(gen_types)
		self.gen_types = gen_types

		self.disc_steps = disc_steps if self.disc is not None else 0
		self.enc_gan = enc_gan and self.enc is not None

		self.criterion = util.get_loss_type(criterion_info) if isinstance(criterion_info, str) else util.get_loss_type(**criterion_info) # TODO: automate

		self.stats.new('reg-vae', 'reg-gan', 'wasserstein', 'reconstruction')
		self.viz_force_gen = viz_force_gen
		self._rec, self._real = None, None

		self.rec_noise = rec_noise

		self.set_optim()

	def _visualize(self, info, logger):
		if self._viz_counter % 2 == 0:
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
			if 'gen' in info:
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


class WPP_RAMMC(WPP_VAE):

	def regularize(self, q):
		pass

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

