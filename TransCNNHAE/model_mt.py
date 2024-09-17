import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from TransCNNHAE.networks import MuliTem_TransCNN
from TransCNNHAE.loss import PerceptualLoss
from model_base import ModelBase
from metrics import PSNR, SSIM, SAM, MAE

from torch.optim import lr_scheduler

import losses

S1_BANDS = 2
S2_BANDS = 13
RGB_BANDS = 3

class ModelCRNet(ModelBase):
    def __init__(self, config):
        super(ModelCRNet, self).__init__()
        self.config = config
        self.net_G = MuliTem_TransCNN(config).cuda()
        self.net_G = nn.DataParallel(self.net_G)
        
        self.optimizer_G = torch.optim.Adam(params=self.net_G.parameters(), lr=config.G_LR, betas=(config.BETA1, config.BETA2))
        
        self.l1_loss = nn.L1Loss()
        self.content_loss = PerceptualLoss()
                        
    def set_input(self, input):
        self.cloudy_data = input['input']['S2'].cuda()
        self.cloudfree_data = input['target']['S2'].cuda()
        self.sar_data = input['input']['S1'].cuda()
        self.cloudy_name = os.path.splitext(os.path.basename(input['input']['S2 path'][0]))[0]

    def forward(self):
        pred_cloudfree_data = self.net_G(torch.cat([self.cloudy_data, self.sar_data], dim=1))
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        self.optimizer_G.zero_grad()

        g_loss = 0
        c_loss = 0
        f_loss = 0

        # g l1 loss ##     
        g_l1_loss = self.l1_loss(self.pred_cloudfree_data, self.cloudfree_data) * self.config.G2_L1_LOSS_WEIGHT
        c_loss = c_loss + g_l1_loss

        # g content loss #
        rgb_ch = np.random.choice(S2_BANDS, RGB_BANDS, replace=False) # false rgb channel
        g_content_loss, g_mrf_loss = self.content_loss(self.pred_cloudfree_data[:, rgb_ch, ...], self.cloudfree_data[:, rgb_ch, ...])
        g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
        g_mrf_loss = g_mrf_loss * self.config.G2_STYLE_LOSS_WEIGHT
        c_loss = c_loss + g_content_loss
        f_loss = f_loss + g_mrf_loss

        g_loss = c_loss + f_loss

        g_loss.backward()
        self.optimizer_G.step()

        # logs = [
        #     ("l_g", g_loss.item()),
        #     ("l_l1", g_l1_loss.item())     
        #     ]

        return g_loss.item()#, logs

    def val_scores(self):
        self.pred_cloudfree_data = self.forward()
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