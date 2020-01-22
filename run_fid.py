import os
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

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate FID')
	parser.add_argument('--model', type=str, required = True, metavar='M',
                    help='path to saved model files to be evaluated')
	parser.add_argument('--save_path', type=str, default = None, metavar='N',
                    help='Path to save result files')
	parser.add_argument('--split', type=str, default = 'all', metavar='S',
                    help='Split to eval Fid on (train/test/full/ all)')
	parser.add_argument('--batch_size', type=int, default = 50, metavar='B',
                    help='Batch size')
	parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
	parser.add_argument('--save-stat', action = 'store_true', help = 'Genrerate and save statistics')
	parser.add_argument('--n_samples', type=int, default=50000, metavar='NS',
                    help='Number of samples to evaluate/train statistics on')
	return parser.parse_args()

def get_stats(model, model_inception, batch_size, n_samples, dataset, mode = True, hybrid = False, stat_ref = False, save_stats = None):
	
	pred_arr = np.empty((n_samples, 2048))
	out = {}
	if stat_ref:
		if dataset == '3dshapes':
			data = datasets.Shapes3D(dataroot = os.environ['FOUNDATION_DATA_DIR'], train = mode, labels = True)
		else:
			raise NotImplementedError
		assert len(data) >= n_samples, 'Number of samples should be less than dataset size'
		indices = np.random.randint(0, len(data), n_samples)
		
	j = 0
	print('Computing Inception Features')
	while j < n_samples:
		if j% (batch_size*100) == 0:
			print('Done', j , 'Samples')
		curr_batch_size = min(batch_size, n_samples - j)
		with torch.no_grad():
			if stat_ref:
				generated, _ = data[indices[j:j+curr_batch_size]]
				generated = generated.to(model.device)
			else:
				if hybrid:
					q = model.sample_prior(curr_batch_size)
					generated = model.decode(util.shuffle_dim(q))
				else:
					generated = model.generate(curr_batch_size)
			
			pred = model_inception(generated)[0]
			if pred.shape[2] != 1 or pred.shape[3] != 1:
				pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

			pred_arr[j:j+curr_batch_size] = pred.cpu().data.numpy().reshape(curr_batch_size, -1)

		j += curr_batch_size
	m = np.mean(pred_arr, axis=0)
	s = np.cov(pred_arr, rowvar=False)
	if save_stats is not None:
		out['indices'] = indices
		out['m'] = m
		out['sigma'] = s
		with open(save_stats,'wb') as b:
			pkl.dump(out, b)
	return m, s

def main(args):
	batch_size = args.batch_size
	block_idx = utils.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
	n_samples = args.n_samples
	path = args.model
	_args, model = train.load(path=path, A=None, get_model=get_model, get_data = None, return_args = True)
	dataset = _args.dataset.name
	model.eval().to(_args.device)
	model_inception = utils.InceptionV3([block_idx]).to(_args.device).eval()
	dataroot = 'data/stats_fid'
	if dataset == '3dshapes':
		file_name = ['3dshapes_stats_fid_train.pkl']
		_train = [True]
		_split = ['Train']
		if args.split == 'train':
			file_name = ['3dshapes_stats_fid_train.pkl']
			_train = [True]
			_split = ['Train']
		elif args.split == 'test':
			file_name = ['3dshapes_stats_fid_test.pkl']
			_train = [False]
			_split = ['Test']
		elif args.split == 'full':
			file_name = ['3dshapes_stats_fid.pkl']
			_train = [None]
			_split = ['Full']
		elif args.split == 'all':
			file_name = ['3dshapes_stats_fid_train.pkl', '3dshapes_stats_fid_test.pkl', '3dshapes_stats_fid.pkl']
			_train = [True, False, None]
			_split = ['Train', 'Test', 'Full']
		else:
			print('Invalid split. Using train split')
	else:
		raise NotImplementedError

	if args.save_stat:
		for id_, i in enumerate(file_name):
			get_stats(model, model_inception, batch_size, n_samples, dataset, mode = _train[id_], stat_ref = True, save_stats = os.path.join(dataroot, dataset, i))
		return 0
	
	m2, s2 = get_stats(model, model_inception, batch_size, n_samples, dataset)
	m2_h, s2_h = get_stats(model, model_inception, batch_size, n_samples, dataset, hybrid = True)

	fid = {}
	fid['Generated'] = {}
	fid['Hybrid'] = {}
	for id_, i in enumerate(file_name):
		if os.path.exists(os.path.join(dataroot, dataset, i)):

			f = pkl.load(open(os.path.join(dataroot, dataset, i),'rb'))
			m1, s1 = f['m'], f['sigma']
		else:
			m1, s1 = get_stats(model, model_inception, batch_size, n_samples, dataset, mode = _train[id_], stat_ref = True)

		fid['Generated'][_split[id_]] = fid_score.calculate_frechet_distance(m1,s1,m2,s2)
		fid['Hybrid'][_split[id_]] = fid_score.calculate_frechet_distance(m1,s1,m2_h,s2_h)

	with open(osp.join(args.save_path, 'results_fid.txt'),'w') as t:
		for i in fid.keys():
			t.write(i + ':\n')
			for j in fid[i].keys():
				t.write('\t'+ j + ':\t' + str(fid[i][j])+'\n')

if __name__ == '__main__':
	args_ = parse_args()
	np.random.seed(args_.seed)
	torch.manual_seed(args_.seed)
	if args_.save_path is None and isinstance(args_.model, str):
		args_.save_path = args_.model
	else:
		args_.save_path = 'temp/'
		print('Storing results in temp')
		if not os.path.exists(args_.save_path):
			os.makedirs(args_.save_path)
	main(args_)
