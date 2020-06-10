import sys
sys.path.append('../Env')
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from CarEnvironment import CarEnvironment

def angular_difference(self, start_config, end_config):
    """ Compute angular difference

        @param start_config: a [3 x n] numpy array of states
        @param end_config: a [3 x 1] numpy array of goal state
    """                
    th1 = start_config[2,:]; th2 = end_config[2,:]
    ang_diff = th1-th2
    ang_diff = ((ang_diff + np.pi) % (2*np.pi)) - np.pi 
    return ang_diff

class MPNetDataset(Dataset):
	def __init__(self):
		folders = os.listdir("../data_rrtstar")
		start = []
		goal = []
		self.next = []
		sample_ratio = 0.1
		for f in folders:
			subfolder = os.path.join("../data_rrtstar", f)
			if not os.path.isdir(subfolder):
				continue

			plans = os.listdir(subfolder)
			for p in plans:
				plan = np.loadtxt(os.path.join(subfolder, p)).T
				size = plan.shape[0]
				for i in range(size-1):
					samples = np.random.choice(size-i-1, int(sample_ratio*(size-i)), replace = False) + (i+1)
					if not (i+1) in samples:
						samples = np.concatenate([[i+1], samples])
					if not (size-1) in samples:
						samples = np.concatenate([samples, [size-1]])
					start.append(plan[i][None].repeat(samples.shape[0], axis = 0))
					goal.append(plan[samples])
					self.next.append(plan[i+1][None].repeat(samples.shape[0], axis = 0))
				# start.append(plan[:-1])
				# goal.append(plan[-1:].repeat(size-1, axis = 0))
				# self.next.append(plan[1:])

		start = np.concatenate(start, axis = 0)[:, :2]
		planning_env = CarEnvironment("../map/map.png")

		self.observe = planning_env.get_measurement(start.T) / 4.
		start = start / np.array([1788, 1240])
		goal = np.concatenate(goal, axis = 0)[:, :2] / np.array([1788, 1240])
		self.start_goal = np.concatenate([start, goal], axis = 1)
		self.next = np.concatenate(self.next, axis = 0)[:, :2] / np.array([1788, 1240])

		self.next = (self.next - start[:, :2]) * 20

	def __len__(self):
		return self.start_goal.shape[0]

	def __getitem__(self, idx):
		return (self.start_goal[idx], self.next[idx], self.observe[idx])

if __name__ == "__main__":
	ds = MPNetDataset()
