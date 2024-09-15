import os
import yaml
import sys
import torch
import argparse
import numpy as np
import random

from data.dataLoader import SEN12MSCR
from model_CR_net import ModelCRNet
from generic_train import Generic_Train
from model_base import print_options, seed_torch

##===================================================##
##********** Configure training settings ************##
##===================================================##

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='TransCNNHAE_test', help='the name of config files')
opts = parser.parse_args()
with open(os.path.join('config', opts.config_name), 'r') as f:
    _yaml = f.read()
    config= yaml.load(_yaml, Loader=yaml.FullLoader)

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch()

dt_train = SEN12MSCR(config.TRAIN_ROOT, split='train')
dt_test = SEN12MSCR(config.TEST_ROOT, split='test')
dt_val = SEN12MSCR(config.VAL_ROOT, split='val')


train_loader = torch.utils.data.DataLoader(
    dt_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)
val_loader = torch.utils.data.DataLoader(
    dt_val,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)
test_loader = torch.utils.data.DataLoader(
    dt_test,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    #num_workers=config.NUM_WORKERS,
)

print("Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test)))

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelCRNet(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
Generic_Train(model, opts, train_loader, val_loader).train()





	