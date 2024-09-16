import os
import torch
import torch.nn as nn
import torch.nn.init as init

from model_base import ModelBase
from metrics import PSNR, SSIM, SAM, MAE

from torch.optim import lr_scheduler

import losses

S1_BANDS = 2
S2_BANDS = 13

def weight_init(m, spread=1.0):
    '''
    Initializes a model's parameters.
    Credits to: https://gist.github.com/jeasinema

    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data, gain=spread)
        if m.bias is not None:
            init.normal_(m.bias.data, mean=0, std=spread)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=0, std=spread)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=spread)
        try:
            init.normal_(m.bias.data, mean=0, std=spread)
        except AttributeError:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data, mean=0, std=spread)

class ModelCRNet(ModelBase):
    def __init__(self, config):
        super(ModelCRNet, self).__init__()
        self.config = config
        
        # create network
        self.print_networks(self.net_G)

        # do random weight initialization
        print('\nInitializing weights randomly.')
        self.net_G.apply(weight_init)
        
        # initialize optimizers
        paramsG = [{'params': self.netG.parameters()}]
        self.optimizer_G = torch.optim.Adam(paramsG, lr=config.LR)

        #scheduler: for G, note: stepping takes place at the end of epoch
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=self.config.gamma)
        
        self.loss_fn = losses.get_loss(self.config)
                        
    def set_input(self, input):
        self.cloudy_data = self.scale_by * input['A'].cuda()
        self.cloudfree_data = self.scale_by * input['B'].cuda()
        self.dates  = None if input['dates'] is None else input['dates'].cuda()
        self.masks  = input['masks'].cuda()
        
    def forward(self):
        pred_cloudfree_data = self.net_G(self.cloudy_data, batch_positions=self.dates)
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        if hasattr(self.net_G, 'vars_idx'):
            self.loss_G, self.netG.variance = losses.calc_loss(self.criterion, self.config, self.fake_B[:, :, :self.netG.mean_idx, ...], self.real_B, var=self.fake_B[:, :, self.netG.mean_idx:self.netG.vars_idx, ...])
        else: # used with all other models
            self.loss_G, self.netG.variance = losses.calc_loss(self.criterion, self.config, self.fake_B[:, :, :S2_BANDS, ...], self.real_B, var=self.fake_B[:, :, S2_BANDS:, ...])

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()  

        return self.loss_G.item()

    def val_scores(self):
        self.pred_cloudfree_data = self.forward()
        scores = {'PSNR': PSNR(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'SSIM': SSIM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'SAM': SAM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'MAE': MAE(self.pred_cloudfree_data.data, self.cloudfree_data),
                  }
        return scores

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.config.SAVE_MODEL_DIR)
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.config.EXP_NAME, self.config.SAVE_MODEL_DIR, '%s_net_CR.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])