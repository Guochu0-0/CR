import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from UnCRtainTS.uncrtaints import UNCRTAINTS
from model_base import ModelBase
from metrics import PSNR, SSIM, SAM, MAE
from UnCRtainTS.losses import get_loss
from torch.optim import lr_scheduler

import UnCRtainTS.losses as losses

S1_BANDS = 2
S2_BANDS = 13
RGB_BANDS = 3

class ModelCRNet(ModelBase):
    def __init__(self, config):
        super(ModelCRNet, self).__init__()
        self.config = config
        self.scale_by = config.SCALE_BY

        if self.config.LOSS == 'MGNLL':
            out_dim = S2_BANDS*2
        else:
            out_dim = S2_BANDS

        self.net_G = UNCRTAINTS(input_dim=config.INPUT_DIM,
                                scale_by=config.SCALE_BY,
                                out_conv=[out_dim],
                                out_nonlin_mean=config.MEAN_NONLINEARITY,
                                out_nonlin_var=config.VAR_NONLINEARITY).cuda()
        self.net_G = nn.DataParallel(self.net_G)
        
        paramsG = [{'params': self.net_G.parameters()}]
        self.optimizer_G = torch.optim.Adam(paramsG, lr=config.LR)

        # 2 scheduler: for G, note: stepping takes place at the end of epoch
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=self.config.GAMMA)
        
        self.loss = get_loss(config)
                        
    def set_input(self, input):
        self.cloudy_data = self.scale_by * torch.stack(input['input']['S2'], dim=1).cuda()
        self.cloudfree_data = self.scale_by * torch.stack(input['target']['S2'], dim=1).cuda()

        self.cloudy_name = os.path.splitext(os.path.basename(input['input']['S2 path'][0][0]))[0]
        in_S2_td    = input['input']['S2 TD']
        if self.config.BATCH_SIZE>1: in_S2_td = torch.stack((in_S2_td)).T

        if self.config.USE_SAR:
            in_S1_td    = input['input']['S1 TD']
            if self.config.BATCH_SIZE>1: in_S1_td = torch.stack((in_S1_td)).T
            self.sar_data = self.scale_by * torch.stack(input['input']['S1'], dim=1).cuda()
            self.input_data = torch.cat([self.sar_data, self.cloudy_data], dim=2)
            self.dates = torch.stack((in_S1_td,in_S2_td)).float().mean(dim=0).cuda()
        else:
            self.input_data = self.cloudy_data
            self.dates = in_S2_td.float().cuda()

    def reset_input(self):
        self.cloudy_data = None
        self.cloudfree_data = None
        self.dates  = None 
        if self.config.USE_SAR: self.sar_data = None

        #self.masks  = None
        del self.cloudy_data
        del self.cloudfree_data 
        del self.dates
        if self.config.USE_SAR: del self.sar_data
        #del self.masks

    def rescale(self):
        # rescale target and mean predictions
        if hasattr(self, 'cloudy_data'): self.cloudy_data = 1/self.scale_by * self.cloudy_data
        self.cloudfree_data = 1/self.scale_by * self.cloudfree_data 
        self.pred_cloudfree_data = 1/self.scale_by * self.pred_cloudfree_data[:,:,:S2_BANDS,...]


    def forward(self):
        pred_cloudfree_data = self.net_G(self.input_data, batch_positions=self.dates)
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        self.optimizer_G.zero_grad()
  
        loss_G, _ = losses.calc_loss(self.loss, self.config, self.pred_cloudfree_data[:, :, :S2_BANDS, ...], self.cloudfree_data, var=self.pred_cloudfree_data[:, :, S2_BANDS:, ...])

        loss_G.backward()
        self.optimizer_G.step()

        # re-scale inputs, predicted means, predicted variances, etc
        self.rescale()
        # resetting inputs after optimization saves memory
        #self.reset_input()

        return loss_G.item()

    def val_scores(self):
        self.pred_cloudfree_data = self.forward()
        self.rescale()
        scores = {'PSNR': PSNR(self.pred_cloudfree_data, self.cloudfree_data),
                  #'SSIM': SSIM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  #'SAM': SAM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  #'MAE': MAE(self.pred_cloudfree_data.data, self.cloudfree_data),
                  }
        return scores
    
    def val_img_save(self, epoch):
        
        imgs_cloudy = []
        for i in range(self.config.INPUT_T):
            cloudy_i = self.cloudy_data[0, i, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
            imgs_cloudy.append(cloudy_i)

        gt = self.cloudfree_data[0, 0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        pred = self.pred_cloudfree_data[0, 0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        sar = self.sar_data[0, 0, [0]].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()

        merged1 = np.concatenate(imgs_cloudy, axis=1)
        merged2 = np.concatenate([sar, pred, gt], axis=1)
        merged = np.concatenate([merged1, merged2], axis=0)

        save_dir = os.path.join('img_gen', self.config.EXP_NAME, f'epoch_{epoch}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(os.path.join(save_dir, f"{self.cloudy_name}.png"), merged) 


    def save_checkpoint(self, epoch):
        self.save_network(self.net_G,  epoch, os.path.join(self.config.SAVE_MODEL_DIR, self.config.EXP_NAME))
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.config.SAVE_MODEL_DIR, self.config.EXP_NAME, '%s_net_CR.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])