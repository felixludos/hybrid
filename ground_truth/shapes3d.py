from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from . import ground_truth_data
from . import utils
import numpy as np
import PIL
from six.moves import range
from foundation import train as trn
from foundation.train import datasets

class Shapes3D(datasets.Shapes3D, ground_truth_data.GroundTruthData):
	"""DSprites dataset.
	The data set was originally introduced in "beta-VAE: Learning Basic Visual
	Concepts with a Constrained Variational Framework" and can be downloaded from
	https://github.com/deepmind/dsprites-dataset.
	The ground-truth factors of variation are (in the default setting):
	0 - shape (3 different values)
	1 - scale (6 different values)
	2 - orientation (40 different values)
	3 - position x (32 different values)
	4 - position y (32 different values)
	"""

	def __init__(self, data_file = os.environ['FOUNDATION_DATA_DIR'], latent_factor_indices=None):
		# By default, all factors (including shape) are considered ground truth
		# factors.
		if latent_factor_indices is None:
		 	latent_factor_indices = list(range(6))
		self.latent_factor_indices = latent_factor_indices
		self.data_shape = [64, 64, 3]

		A = trn.get_config()
		A.dataroot = data_file
		A.train = None
		A.labeled = True
		super().__init__(A)

		# super().__init__(dataroot = data_file, train = None, labels = True)
		self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.latent_factor_indices = list(range(6))
		self.num_total_factors = self.labels.shape[1]
		self.state_space = utils.SplitDiscreteStateSpace(self.factor_sizes,
		                                                self.latent_factor_indices)
		self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
		    self.factor_sizes)

	@property
	def num_factors(self):
		return self.state_space.num_latent_factors

	@property
	def factors_num_values(self):
		return self.factor_sizes

	@property
	def observation_shape(self):
		return [64, 64, 3]

	def sample_factors(self, num, random_state):
		"""Sample a batch of factors Y."""
		return self.state_space.sample_latent_factors(num, random_state)

	def sample_observations_from_factors(self, factors, random_state):
		all_factors = self.state_space.sample_all_factors(factors, random_state)
		indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
		images, labels = self[indices]
		return images