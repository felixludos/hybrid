import os
import os.path as osp
import numpy as np
from metrics import fid_score
from metrics import metric_utils as utils
import h5py as hf
import torch
from foundation import train, util
from hybrid import get_model
from torch.nn.functional import adaptive_avg_pool2d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, required = True)
args_ = parser.parse_args()
print(args_.i)

"""
models_list = ['wae_191204-215052',
 'vae_191204-214838',
 'vaep01_191204-215053',
 'fvae_191204-214742',
 'fdwae1e-4-probp5_191204-214751',
 'fwae_191204-214848',
 'dwae1e-4-probp5_191205-145820',
 'fdwae1e-4-probp5-priorp2_191205-145821',
 'wpp1e-4_191204-214741',
 'gan-disclr3x_191204-214753',
 'wpp5e-5_191204-214751',
 'fdvae1e-4-probp5_191205-145826',
 'fdvae1e-4-probp2_191205-145831',
 'gan_191205-145831']
 """
i = args_.i
n_samples = 480000
batch_size = 50
block_idx = utils.InceptionV3.BLOCK_INDEX_BY_DIM[2048]

model_inception = utils.InceptionV3([block_idx]).to('cuda').eval()
f = np.load('data/3dshapes_stats_fid.npz', allow_pickle = True)['arr_0'][()]
m1, s1 = f['mu'][:], f['sigma'][:]
path = osp.join('saved_models/shapes3d',i)
out_path = osp.join('results_3dshapes',i)
args, model = train.load(path=path, A=None, get_model=get_model)

model.eval().to(args['device'])
j = 0
pred_arr = np.empty((n_samples, 2048))
while j < n_samples:
	print(j)
	curr_batch_size = min(batch_size, n_samples - j)
	#generated = model.generate(curr_batch_size)
	q = model.sample_prior(curr_batch_size)
	generated = model.decode(util.shuffle_dim(q))
	pred = model_inception(generated)[0]
	if pred.shape[2] != 1 or pred.shape[3] != 1:
		pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

	pred_arr[j:j+curr_batch_size] = pred.cpu().data.numpy().reshape(curr_batch_size, -1)

	j += curr_batch_size

m2 = np.mean(pred_arr, axis=0)
s2 = np.cov(pred_arr, rowvar=False)
fid = fid_score.calculate_frechet_distance(m1,s1,m2,s2)
with open(osp.join(out_path,'results_fid_hybrid.txt'),'w') as t:
	t.write('fid score hybrid:\t' + str(fid))




