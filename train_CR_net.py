import os
import yaml
import sys
import torch
import argparse
import numpy as np
from config import Config

from data.dataLoader import SEN12MSCR, SEN12MSCRTS
from UnCRtainTS.model_CR_net import ModelCRNet
from generic_train import Generic_Train
from model_base import print_options, seed_torch

##===================================================##
##********** Configure training settings ************##
##===================================================##

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='UnCRtainTS_baseline', help='the name of config files')
opts = parser.parse_args()
config = Config(os.path.join('config', f'{opts.config_name}.yml'))

##===================================================##
##****************** choose gpu *********************##
##===================================================##
#os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch()

if config.DATASET == 'sen12mscr':
    dt_train = SEN12MSCR(config.TRAIN_ROOT, split='train')
    dt_val = SEN12MSCR(config.VAL_ROOT, split='val')
elif config.DATASET == 'sen12mscrts':
    dt_train = SEN12MSCRTS(config.TRAIN_ROOT, 
                           split='train', 
                           min_cov=config.MIN_COV,
                           max_cov=config.MAX_COV,
                           n_input_samples=config.INPUT_T,
                           sampler=config.SAMPLE_TYPE, 
                           import_data_path=config.TRAIN_IMPORT_PATH)
    dt_val = SEN12MSCRTS(config.VAL_ROOT, 
                         split='val', 
                         min_cov=config.MIN_COV,
                         max_cov=config.MAX_COV,
                         n_input_samples=config.INPUT_T,
                         sampler=config.SAMPLE_TYPE, 
                         import_data_path=config.VAL_IMPORT_PATH)

if config.SUB_NUM:
    dt_train = torch.utils.data.Subset(dt_train, np.random.choice(len(dt_train), config.SUB_NUM, replace=False))
    dt_val = torch.utils.data.Subset(dt_val, np.random.choice(len(dt_train), config.SUB_NUM, replace=False))

train_loader = torch.utils.data.DataLoader(
    dt_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    dt_val,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

print("Train {}, Val {}".format(len(dt_train), len(dt_val)))

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelCRNet(config)

##===================================================##
##**************** Train the network ****************##
##===================================================##
Generic_Train(model, config, train_loader, val_loader).train()
