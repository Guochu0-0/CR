import os
import time
import torch
import numpy as np
import tqdm
from data.dataPrepare import prepare_data

class Generic_Train():
	def __init__(self, model, opts, train_dataloader, val_dataloader):
		self.model=model
		self.opts=opts
		self.train_dataloader=train_dataloader
		self.val_dataloader=val_dataloader

	def train(self):
		total_steps = 0
		log_loss = 0
		best_score = 0

		for epoch in range(self.opts.epochs):
			for i, batch in enumerate(tqdm(self.train_dataloader)):
				total_steps+=1

				device = torch.device(self.opts.device)
				if self.opts.sample_type == 'cloudy_cloudfree':
					x, y, in_m, dates = prepare_data(batch, device, self.opts)
				elif self.opts.sample_type == 'pretrain':
					x, y, in_m = prepare_data(batch, device, self.opts)
					dates = None
				else:
					raise NotImplementedError
				input = {'A': x, 'B': y, 'dates': dates, 'masks': in_m}
				
				self.model.set_input(input)
				batch_loss = self.model.optimize_parameters()
				log_loss = log_loss + batch_loss

				if total_steps % self.opts.log_iter == 0:
					avg_log_loss = log_loss/self.opts.log_iter
					print('epoch', epoch, 'steps', total_steps, 'loss', avg_log_loss)
					log_loss = 0

			if (epoch+1) % self.opts.val_freq == 0:
				print("validation...")
				self.model.net_G.eval()
				with torch.no_grad():
					_iter = 0
					score = 0
					for data in self.val_dataloader:
						device = torch.device(self.opts.device)
						if self.opts.sample_type == 'cloudy_cloudfree':
							x, y, in_m, dates = prepare_data(data, device, self.opts)
						elif self.opts.sample_type == 'pretrain':
							x, y, in_m = prepare_data(data, device, self.opts)
							dates = None
						else:
							raise NotImplementedError
						input = {'A': x, 'B': y, 'dates': dates, 'masks': in_m}
						self.model.set_input(input)
						score += self.model.val_scores()['PSNR']
						_iter += 1
					score = score/_iter
				print(f'PSNR: {score}')
				if score > best_score:  # save best model
					best_score = score
					self.model.save_checkpoint('best')
				self.model.net_G.train()
			
			self.model.scheduler_G.step()
			
			if epoch % self.opts.save_freq == 0:
				self.model.save_checkpoint(epoch)



