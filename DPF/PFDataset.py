import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class PFDataset(Dataset):
	'''dataset for training DPF'''
	def __init__(self):
		num_traj = len(os.listdir("../data_rrt/")) // 3
		actions = []
		measurements = []
		states = []
		next_states = []
		for i in range(200):
			action = np.loadtxt("../data_rrt/action_%d.txt" % i)
			measurement = np.loadtxt("../data_rrt/measure_%d.txt" % i)
			rollout = np.loadtxt("../data_rrt/rollout_%d.txt" % i)

			actions.append(action.T)
			measurements.append(measurement[1:])
			states.append(rollout[:, :-1].T)
			next_states.append(rollout[:, 1:].T)

		self.actions = np.concatenate(actions, axis = 0)
		self.measurements = np.concatenate(measurements, axis = 0)
		self.states = np.concatenate(states, axis = 0)
		self.next_states = np.concatenate(next_states, axis = 0)
		self.deltas = (self.next_states - self.states)
		self.deltas[:, 2][self.deltas[:, 2] >  np.pi] =  self.deltas[:, 2][self.deltas[:, 2] > np.pi]  - 2*np.pi
		self.deltas[:, 2][self.deltas[:, 2] < -np.pi] =  self.deltas[:, 2][self.deltas[:, 2] < -np.pi] + 2*np.pi
		
		# scaling
		self.actions = self.actions / np.array([20., 1.0]) # [-1, 1]
		self.measurements = self.measurements / 4.0 # [0, 1]
		self.states = self.states / np.array([1788, 1240, 1.0]) # [0, 1]
		self.next_states = self.next_states / np.array([1788, 1240, 1.0]) # [0, 1]
		self.deltas = self.deltas / np.array([2, 2, 20/7.5*np.tan(1.0)*0.1]) # [-1, 1]

		# downsample
		self.states = self.states[::10]
		self.actions = self.actions[::10]
		self.measurements = self.measurements[::10]
		self.next_states = self.next_states[::10]
		self.deltas = self.deltas[::10]

	def __len__(self):
		return self.actions.shape[0]

	def __getitem__(self, idx):
		return (self.states[idx], self.actions[idx], self.measurements[idx], self.next_states[idx], self.deltas[idx])

if __name__ == "__main__":
	dataset = PFDataset()
