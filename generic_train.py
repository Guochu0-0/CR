import torch
import time
import os
import numpy as np
from tqdm import tqdm
from metrics import PSNR
from tensorboardX import SummaryWriter

class Generic_Train():
	def __init__(self, model, config, train_dataloader, val_dataloader):
		self.model=model
		self.config=config
		self.train_dataloader=train_dataloader
		self.val_dataloader=val_dataloader

		self.writer = SummaryWriter(os.path.join('runs', config.EXP_NAME))

	def train(self):
		
		total_steps = 0
		log_loss = 0
		best_score = 0

		for epoch in range(self.config.EPOCH):

			train_psnr = 0

			time_1 = time.time()
			for i, batch in enumerate(tqdm(self.train_dataloader)):
				total_steps+=1

				time_2 = time.time()
				time_load = time_2 - time_1
				print(f"time_load: {time_load}s")

				self.model.set_input(batch)
				batch_loss = self.model.optimize_parameters()
				log_loss = log_loss + batch_loss

				train_psnr += PSNR(self.model.pred_cloudfree_data, self.model.cloudfree_data) 

				if total_steps % self.config.LOG_ITER == 0:

					avg_log_loss = log_loss/self.config.LOG_ITER

					self.writer.add_scalar('batch_loss', avg_log_loss, total_steps)

					log_loss = 0
				
				time_1 = time.time()
				time_process = time_1 - time_2
				print(f"time_process: {time_process}s")

			self.writer.add_scalar('train_psnr', train_psnr/len(self.train_dataloader), epoch)

			if (epoch+1) % self.config.VAL_FREQ == 0:

				print("validation...")
				score = self.val(epoch)
				self.writer.add_scalar('val_psnr', score, epoch)

				if score > best_score:  # save best model
					best_score = score
					self.model.save_checkpoint('best')
			
			#self.model.scheduler_G.step()
			
			if epoch % self.config.SAVE_FREQ == 0:
				self.model.save_checkpoint(epoch)


	def val(self, epoch):
		self.model.net_G.eval()

		with torch.no_grad():

			_iter = 0
			score = 0

			for data in self.val_dataloader:
				self.model.set_input(data)
				score += self.model.val_scores()['PSNR']
				_iter += 1

				if _iter%2 == 0:
					self.model.val_img_save(epoch)

			score = score/_iter
		
		self.model.net_G.train()

		return score

