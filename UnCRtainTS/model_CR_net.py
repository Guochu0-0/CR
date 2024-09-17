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
        self.scaly_by = config.SCALE_BY
        self.net_G = UNCRTAINTS(input_dim=config.INPUT_DIM,
                                scale_by=config.SCALE_BY,
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
        self.cloudfree_data = self.scale_by * input['target']['S2'].cuda()

        self.cloudy_name = os.path.splitext(os.path.basename(input['input']['S2 path'][0][0]))[0]
        in_S2_td    = input['input']['S2 TD']

        if self.config.USE_SAR:
            in_S1_td    = input['input']['S1 TD']
            self.sar_data = self.scale_by * torch.stack(input['input']['S1'], dim=1).cuda()
            self.input_data = torch.stack([self.sar_data, self.cloudy_data], dim=2)
            self.dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).cuda()
        else:
            self.input_data = self.cloudy_data
            self.dates = torch.tensor(in_S2_td).float().cuda()

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
        
        # rescale (co)variances
        if hasattr(self.net_G, 'variance') and self.net_G.variance is not None:
            self.net_G.variance = 1/self.scale_by**2 * self.net_G.variance


    def forward(self):
        pred_cloudfree_data = self.net_G(self.cloudy_data, batch_positions=self.dates)
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        self.optimizer_G.zero_grad()
  
        loss_G, self.net_G.variance = losses.calc_loss(self.loss, self.config, self.pred_cloudfree_data[:, :, :self.net_G.mean_idx, ...], self.cloudfree_data, var=self.pred_cloudfree_data[:, :, self.net_G.mean_idx:self.net_G.vars_idx, ...])

        loss_G.backward()
        self.optimizer_G.step()

        # re-scale inputs, predicted means, predicted variances, etc
        self.rescale()
        # resetting inputs after optimization saves memory
        self.reset_input()

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
        
        cloudy = self.cloudy_data[0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        gt = self.cloudfree_data[0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        pred = self.pred_cloudfree_data[0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        sar = self.sar_data[0, [0]].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()

        merged = np.concatenate([cloudy, sar, pred, gt], axis=1)

        save_dir = os.path.join('img_gen', self.config.EXP_NAME, f'epoch_{epoch}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(os.path.join(save_dir, f"{self.cloudy_name}.png"), merged) 


    def save_checkpoint(self, epoch):
        self.save_network(self.net_G,  epoch, os.path.join(self.config.SAVE_MODEL_DIR, self.config.EXP_NAME))
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.config.SAVE_MODEL_DIR, self.config.EXP_NAME, '%s_net_CR.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])