import sys
sys.path.append('../Env')
import torch
import torch.nn as nn
import numpy as np

from MPNetDataset import MPNetDataset
from torch.utils.data import DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPBlock(nn.Module):
	def __init__(self, inputDim, outputDim, use_drop = True):
		super(MLPBlock, self).__init__()
		self.fc = nn.Linear(inputDim, outputDim)
		self.activate = nn.PReLU()
		self.m = nn.BatchNorm1d(outputDim)
		self.use_drop = use_drop
		if self.use_drop:
			self.drop = nn.Dropout(p=0.5)

	def forward(self, x):
		if self.use_drop:
			return self.drop(self.m(self.activate(self.fc(x))))
		else:
			return self.m(self.activate(self.fc(x)))

# class ResidualBlock(nn.Module):
# 	def __init__(self, inputDim, outputDim):
# 		super(ResidualBlock, self).__init__():
# 		self.fc1 = nn.Linear(inputDim, )

class MPNet():
	def __init__(self, state_dim = 2):
		self.planner = nn.Sequential(MLPBlock(128, 256),
									 MLPBlock(256, 128),
									 MLPBlock(128, 64),
									 #MLPBlock(256, 256),
									 #MLPBlock(256, 128),
									 #MLPBlock(128, 64),
									 MLPBlock(64, 32),
									 nn.Linear(32, state_dim)).to(device)
		self.planner_optimizer = torch.optim.Adam(self.planner.parameters(), lr = 0.001)

		self.obs_encoder = nn.Sequential(MLPBlock(180, 256),
										 MLPBlock(256, 64)).to(device)
		self.obs_encoder_optimizer = torch.optim.Adam(self.obs_encoder.parameters(), lr = 0.001)

		self.state_encoder = nn.Sequential(MLPBlock(4, 32),
										   MLPBlock(32, 64)).to(device)
		self.state_encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr = 0.001)

	def forward(self, state, obs):
		obs_encode = self.obs_encoder(obs)
		state_encode = self.state_encoder(state)
		delta = self.planner(torch.cat([state_encode, obs_encode], dim = 1))
		return delta

	def train(self, loader, max_iter = 100):
		mseLoss = nn.L1Loss()
		#mseLoss = nn.MSELoss()
		print("start training...")
		for it in range(max_iter):
			total_loss = []
			for _, (start_goal, next_state, obs) in enumerate(loader):
				start_goal = start_goal.float().to(device)
				next_state = next_state.float().to(device)
				obs = obs.float().to(device)

				pred = self.forward(start_goal, obs)
				loss = mseLoss(pred, next_state)

				self.planner_optimizer.zero_grad()
				self.obs_encoder_optimizer.zero_grad()
				self.state_encoder_optimizer.zero_grad()
				loss.backward()
				self.planner_optimizer.step()
				self.obs_encoder_optimizer.step()
				self.state_encoder_optimizer.step()

				total_loss.append(loss.detach().cpu().numpy())
			if it % 20 == 0:
				self.save()
			print("epoch: %d, loss: %2.8f" % (it, np.mean(total_loss)))

	def predict(self, x, obs):
		x = torch.FloatTensor(x).to(device)
		obs = torch.FloatTensor(obs).to(device)
		self.planner.eval()
		self.obs_encoder.eval()
		self.state_encoder.eval()
		with torch.no_grad():
			return self.forward(x, obs).cpu().numpy()

	def save(self):
		if not os.path.exists("weights/"):
			os.mkdir("weights/")
		file_name = "weights/MPNet1.pt"
		torch.save({"planner" : self.planner.state_dict(),
					"obs_encoder" : self.obs_encoder.state_dict(),
					"state_encoder" : self.state_encoder.state_dict()}, file_name)
		print("save model to " + file_name)

	def load(self):
		try:
			if not os.path.exists("weights/"):
				os.mkdir("weights/")
			file_name = "weights/MPNet1.pt"
			checkpoint = torch.load(file_name)
			self.planner.load_state_dict(checkpoint["planner"])
			self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
			self.state_encoder.load_state_dict(checkpoint["state_encoder"])
			print("load model from " + file_name)
		except:
			print("fail to load model!")

if __name__ == "__main__":
	dataset = MPNetDataset()
	loader = DataLoader(dataset, batch_size = 128, shuffle = True, num_workers=4)

	mpnet = MPNet()
	#mpnet.load()
	mpnet.train(loader, 200)
	mpnet.save()
	