import os
import sys

# 添加项目根目录到sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import sys
sys.path.append("../..")
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
print(torch.cuda.is_available())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from baselines.models import TimesNet, DLinear, PatchTST, MICN, \
	iTransformer, TimeMixer, MambaSimple, Pathformer, MSGNet 
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

parser = argparse.ArgumentParser('ITS Forecasting')
parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")

parser.add_argument('--epoch', type=int, default=100, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")

parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--viz', action='store_true', help="Show plots while training")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='ushcn', help="Dataset to load. Available: physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='iTransformer', help="[iTransformer, ...... ]")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('--patch_ts', type=bool, default=False)

parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--alpha', type=float, default=0.9, help="Uncertainty base number")
parser.add_argument('--res', type=float, default=1, help="Res")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

# add seq_len,top_k,nums_of_kernels
parser.add_argument('--seq_len', type=int, default=96, help="Sequence length for the model")
parser.add_argument('--top_k', type=int, default=5, help='Top K value for TimesNet')
parser.add_argument('--num_kernels', type=int, default=64, help='Number of kernels for TimesNet')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn as nn
import torch.optim as optim
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
import lib.utils as utils
from lib.plotting import *
from lib.parse_datasets import parse_datasets
from model.patchmtgnn import *
from baselines.models import *
import warnings
warnings.filterwarnings("ignore")

file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()

print("PID, device:", args.PID, args.device)

#####################################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, patch_ts=args.patch_ts)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
	args.patch_layer = layer_of_patches(args.npatch)

	if args.dataset == 'activity':
		if args.history == 3000:
			args.seq_len = 98
		elif args.history == 2000:
			args.seq_len = 65
		elif args.history == 1000:
			args.seq_len = 34

	elif args.dataset == 'physionet':
		if args.history == 24:
			args.seq_len = 128
		elif args.history == 12:
			args.seq_len = 83
		elif args.history == 36:
			args.seq_len = 175

	elif args.dataset == 'mimic':
		if args.history == 24:
			args.seq_len = 280
		elif args.history == 12:
			args.seq_len = 133
		elif args.history == 36:
			args.seq_len = 464

	elif args.dataset == 'ushcn':
		args.seq_len = 205


	if args.model == 'iTransformer': # add iTransformer
		args.task_name = 'long_term_forecast'
		args.pred_len =  720
		args.output_attention = None
		args.d_model = 512
		args.embed = 'timeF'
		args.freq = 'h'
		args.dropout = 0.1
		args.factor = 3
		args.d_ff = 512
		args.e_layers = 4
		args.n_heads = 1
		args.activation = 'gelu'

	elif args.model == 'DLinear': # add DLinear
		args.task_name = 'long_term_forecast'
		args.moving_avg = 25 # moving_avg in Autoformer
		args.pred_len =  720 # pred_len
		# args.output_attention = None 
		args.individual = False
		# args.features = 'M'
		# args.label_len = 48
		# args.e_layers = 2
		# args.d_layers = 1
		# args.factor = 3
		args.enc_in = 321 # enc_in
		# args.dec_in = 321 
		args.c_out = 321

	elif args.model == 'TimesNet':
		args.task_name = 'long_term_forecast' # task_name
		args.pred_len =  720 # pred_len
		# args.seq_len = 96
		# args.features = 'M'
		args.label_len = 48 # label_len
		# args.output_attention = None
		args.d_model = 64 # d_model
		args.embed = 'timeF' # embed
		args.freq = 'm' # freq
		args.dropout = 0.1 # dropout
		# args.factor = 3
		args.d_ff = 64 # d_ff
		args.num_kernels = 6 # nums_kernels
		args.e_layers = 2 # e_layers
		# args.d_layers = 1
		#### change
		args.enc_in = 12 # enc_in change
		# args.dec_in = 8
		args.c_out = 8 # c_out
		# args.n_heads = 1
		args.top_k = 5 # top_k
	
	elif args.model == 'PatchTST':
		args.task_name = 'long_term_forecast'  # task_name
		# args.is_training = 1  
		args.label_len = 48  # label_len
		args.pred_len = 96  # pred_len
		args.patch_len = 16 # patch_len
		args.stride = 8 # stride
		args.d_model = 64 # d_model
		args.dropout = 0.1 # drop_out
		args.output_attention = None # output_attention
		args.n_heads = 1 # n_head
		args.d_ff = 64 # d_ff
		args.activation = 'gelu' # activation
		args.e_layers = 2  # e_layers
		# args.d_layers = 1  # d_layers
		args.factor = 3  # factor
		args.enc_in = 321  # enc_in
		# args.dec_in = 321  # dec_in
		# args.c_out = 321  # c_out

	elif args.model == 'Pathformer':
		args.task_name = 'long_term_forecast'  # task_name
		args.layer_nums = 3
		args.pred_len = 96  # pred_len
		args.num_nodes = 321
		args.pre_len = 96
		args.k = 2
		args.d_model = 16
		args.d_ff = 64
		args.residual_connection = 1
		args.revin = 1
		args.num_experts_list = [4, 4, 4]
		args.patch_size_list = [16,12,8,32,12,8,6,4,8,6,4,2]

	elif args.model == 'TimeMixer':
		args.task_name = 'long_term_forecast' 
		# args.seq_len = 96  
		args.label_len = 48
		args.pred_len = 96 
		args.e_layers = 2  
		args.moving_avg = 25
		args.enc_in = 7
		# args.dec_in = 321  # dec_in
		args.c_out = 7
		args.d_model = 16  
		args.embed = 'timeF'
		args.freq = 'h'
		args.dropout = 0.1
		args.d_ff = 32  
		args.down_sampling_layers = 0 
		args.down_sampling_method = 'avg'  
		args.down_sampling_window = 1 
		args.channel_independence = 1
		args.use_norm = 1
	
	elif args.model == 'MSGNet':
		args.task_name = 'long_term_forecast' 
		args.freq = 'h' # freq
		args.seasonal_patterns = 'Monthly' 
		args.seq_len = 96 # seq_len
		args.label_len = 48  # label
		args.pred_len = 96  # pred_len
		args.e_layers = 2 # e_layers
		args.enc_in = 7 # enc_in
		args.c_out = 7 # c_out
		args.d_model = 512 # d_model
		args.d_ff = 64 # d_ff
		args.n_heads = 8 # head
		args.dropout = 0.1 # dropout
		args.top_k = 5 # top_k
		args.nums_kernels = 6 # nums_kernels
		args.conv_channel = 32 # conv_channel
		args.skip_channel = 32 # skip
		args.gcn_depth = 2 # 
		args.propalpha = 0.3 #
		args.node_dim = 10 # 
		args.gcn_depth = 2
		args.gcn_dropout = 0.3
		args.embed = 'timeF' # 
		args.individual = False # individual
		args.num_nodes = 7
		args.subgraph_size = 3
		args.tanhalpha = 3
	elif args.model == 'MICN':
		args.task_name = 'long_term_forecast'  # task_name
		# args.is_training = 1  # is_training
		# args.root_path = './dataset/electricity/'  # root_path
		# args.data_path = 'electricity.csv'  # data_path
		# args.model_id = 'ECL_96_96'  # model_id
		# args.data = 'custom'  # data
		args.features = 'M'  # features
		args.seq_len = 96  # seq_len
		args.label_len = 96  # label_len
		args.pred_len = 96  # pred_len
		args.e_layers = 2  # e_layers
		args.d_layers = 1  # d_layers
		args.factor = 3  # factor
		args.enc_in = 321  # enc_in
		args.dec_in = 321  # dec_in
		args.c_out = 321  # c_out
		args.d_model = 256  # d_model
		args.d_ff = 512  # d_ff
		args.top_k = 5  # top_k
		args.des = 'Exp'  # des
		args.itr = 1  # itr
		args.conv_kernel = 24
		
	

	model_dict = {
		# 'TimesNet': TimesNet.Model({  'seq_len': 96 ,
		# 					  'label_len': 48 ,
		# 					  'pred_len': 96 ,
		# 					  'e_layers': 2 ,
		# 					  'd_layers': 1 ,
		# 					  'factor': 3 ,
		# 					  'enc_in': 21 ,
		# 					  'dec_in': 21 ,
		# 					  'c_out': 21 ,
		# 					  'd_model': 32 ,
		# 					  'd_ff': 32 ,
		# 					  'top_k': 5 ,
		# }),
		# 'TimesNet': TimesNet.Model(args),
		# 'DLinear': DLinear.Model(args),
	    'iTransformer': iTransformer.Model(args),
		# 'TimesNet': TimesNet.Model(args),
		# 'MambaSimple': MambaSimple.Model(args),
		# 'TimeMixer': TimeMixer.Model(args),
		# 'Pathformer': Pathformer.Model(args),
		# 'PatchTST': PatchTST.Model(args)
		# 'MSGNet': MSGNet.Model(args),
		# 'MICN': MICN.Model(args),
	}
	# model = Pathformer(args).to(args.device)
	# model = BaselineHPG(args).to(args.device)
	model = model_dict[args.model].to(args.device)


	params = (list(model.parameters()))
	print('model', model)
	print('parameters:', count_parameters(model))
	##################################################################

	# if args.viz:
	# 	viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr_{}seed.log". \
			format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr, args.seed)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	for itr in range(args.epoch):
		st = time.time()
		# max_len = 0
		### Training ###
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

			# max_len = batch_dict['observed_tp'].shape[-1] if batch_dict['observed_tp'].shape[-1] > max_len else max_len
			# print(batch_dict["tp_to_predict"].shape, batch_dict["observed_data"].shape, \
		 	# 	batch_dict["observed_tp"].shape, batch_dict["observed_mask"].shape)
			train_res = compute_all_losses(model, batch_dict)
			train_res["loss"].backward()
			optimizer.step()
		# print(args.dataset + ' ' + str(args.history) + ' max_len_1:', max_len)
		### Validation ###
		model.eval()
		with torch.no_grad():

			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			if(val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			
			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
			 		 test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))
'''
			if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			sys.exit(0)
'''
		


