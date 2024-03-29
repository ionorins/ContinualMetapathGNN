import argparse
import torch
import os
import sys

sys.path.append('..')
from graph_recsys_benchmark.models import PEAGATRecsysModel
from graph_recsys_benchmark.utils import get_folder_path, update_pea_graph_input
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'Graph'
LOSS_TYPE = 'BPR'
MODEL = 'PEAGAT'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')		#Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')	#25m, latest-small
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='unseen', help='') # unseen(for latest-small), random(for Yelp,25m)
parser.add_argument('--entity_aware', type=str, default='false', help='')

# Model params
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--num_heads', type=int, default=1, help='')
parser.add_argument('--repr_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--meta_path_steps', type=str, default='2,2,2,2,2,2,2', help='')	#2,2,2,2,2,2,2,2,2(for small) #2,2,2,2,2,2,2,2,2,2,2,2,2(for 25m) #2,2,2,2,2,2,2,2,2,2,2 (for yelp)
parser.add_argument('--channel_aggr', type=str, default='att', help='')
parser.add_argument('--entity_aware_coff', type=float, default=0.1, help='')

# Train params
parser.add_argument('--init_eval', type=str, default='false', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=1, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=1, help='')             #5(for MovieLens), 3(for Yelp)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for MovieLens), 20(only for Yelp)
parser.add_argument('--batch_size', type=int, default=1024, help='')    #1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='10000,20000', help='')
parser.add_argument('--save_every_epoch', type=int, default=10000, help='')        #26(for MovieLens), 16(only for Yelp)
parser.add_argument('--metapath_test', type=str, default='false', help='')
parser.add_argument('--continual_aspect', type=str, default='continual', help='')
parser.add_argument('--equal_timespan_timeframes', type=str, default='false', help='')
parser.add_argument('--num_timeframes', type=int, default=25, help='')
parser.add_argument('--end_timeframe', type=int, help='')
parser.add_argument('--start_timeframe', type=int, default=0, help='')
parser.add_argument('--ewc_type', type=str, default='ewc', help='')
parser.add_argument('--ewc_lambda', type=float, default=0, help='')
parser.add_argument('--theta', type=float, default=1.5, help='')
parser.add_argument('--future_testing', type=str, default='false', help='')
parser.add_argument('--last_task_accuracy', type=str, default='false', help='')
parser.add_argument('--train_between_emb_diff', type=str, default='false', help='')
parser.add_argument('--out_filename', type=str, default='', help='')

args = parser.parse_args()

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)
    
# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': args.if_use_features.lower() == 'true', 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'cf_loss_type': LOSS_TYPE, 'type': GRAPH_TYPE,
    'sampling_strategy': args.sampling_strategy, 'entity_aware': args.entity_aware.lower() == 'true',
    'model': MODEL,
    'equal_timespan_timeframes': args.equal_timespan_timeframes.lower() == 'true', 'num_timeframes': args.num_timeframes, 
    'end_timeframe': args.end_timeframe, 'start_timeframe': args.start_timeframe,
    'continual_aspect': args.continual_aspect,
    'future_testing': args.future_testing.lower() == 'true',
    'last_task_accuracy': args.last_task_accuracy.lower() == 'true',
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features.lower() == 'true',
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout,
    'num_heads': args.num_heads, 'meta_path_steps': [int(i) for i in args.meta_path_steps.split(',')],
    'channel_aggr': args.channel_aggr,
    'entity_aware': args.entity_aware.lower() == 'true',
    'entity_aware_coff': args.entity_aware_coff,
    'ewc_type': args.ewc_type, 'ewc_lambda': args.ewc_lambda,
}
path_args = model_args.copy()
path_args['meta_path_steps'] = len(path_args['meta_path_steps'])
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,  'device': device,
    'lr': args.lr,
    'num_workers': args.num_workers,
    'weights_folder': os.path.join(weights_folder, str(path_args)[:255]),
    'logger_folder': os.path.join(logger_folder, str(path_args)[:255]),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch,
    'metapath_test': args.metapath_test.lower() == 'true',
    'theta': args.theta,
    'train_between_emb_diff': args.train_between_emb_diff.lower() == 'true',
    'out_filename': args.out_filename,
}

print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


class PEAGATRecsysModel(PEAGATRecsysModel):
    def update_graph_input(self, dataset):
        return update_pea_graph_input(dataset_args, train_args, dataset)


if __name__ == '__main__':
    solver = BaseSolver(PEAGATRecsysModel, dataset_args, model_args, train_args)
    solver.run()
