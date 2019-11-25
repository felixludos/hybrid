# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from . import ground_truth_data
from . import utils
import numpy as np
import PIL
from six.moves import range



class DSprites(ground_truth_data.GroundTruthData):
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

  def __init__(self, data_file = '../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', latent_factor_indices=None):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    if latent_factor_indices is None:
      latent_factor_indices = list(range(5))
    self.latent_factor_indices = latent_factor_indices
    self.data_shape = [64, 64, 1]
    # Load the data so that we can sample from it.
      # Data was saved originally using python2, so we need to set the encoding.
    data = np.load(data_file, encoding="latin1", allow_pickle=True)
    self.images = np.array(data["imgs"])
    self.factor_sizes = np.array(
          data["metadata"][()]["latents_sizes"], dtype=np.int64)
    self.factor_sizes = self.factor_sizes[1:]
    self.full_factor_sizes = [3, 6, 40, 32, 32]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)
    self.state_space = utils.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)