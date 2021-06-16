import os
from run import RunLookalike
import torch
import numpy as np
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='WD')
parser.add_argument('--sample_method', default='normal')#unit sqrt normal
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--task_count', type=int, default=5)
parser.add_argument('--num_expert', type=int, default=8)
parser.add_argument('--num_output', type=int, default=5)
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--embed_size', type=int, default=64)
parser.add_argument('--local_train_lr', type=float, default=0.0002)
parser.add_argument('--local_test_lr', type=float, default=0.001)
parser.add_argument('--global_lr', type=float, default=0.001)
args = parser.parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print('local train lr: {}; local test lr: {}; global lr: {}; epoch: {}; gpu:{}'.format(args.local_train_lr, args.local_test_lr, args.global_lr, args.epoch, args.gpu))

config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'root_path': './data/processed_data/',
        'is_meta': False,
        'weight_decay': 0,
        'model':
            {
            'mlp': {'dims': (64, 64), 'dropout': 0.2}
            },
        'emb_dim': args.embed_size,
        'local_train_lr': args.local_train_lr,
        'local_test_lr': args.local_test_lr,
        'global_lr': args.global_lr,
        'epoch': args.epoch,
        'wd': 0,
        'sample_method': args.sample_method,
        'num_expert': args.num_expert,
        'num_output': args.num_output,
        'task_count': args.task_count
        }

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    RunLookalike(config = config,
        base_model_name = args.model_name,
        ).main()
