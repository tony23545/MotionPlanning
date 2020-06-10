import numpy as np

class ReplayBuffer():
	def __init__(self, max_size):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def push(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size = batch_size)
		obs, local_goal, next_obs, next_goal, action, reward, done = [], [], [], [], [], [], []
		for i in ind:
			obs_, local_goal_, next_obs_, next_goal_, action_, reward_, done_ = self.storage[i]
			obs.append(np.array(obs_, copy = False))
			local_goal.append(np.array(local_goal_, copy = False))
			next_obs.append(np.array(next_obs_, copy = False))
			next_goal.append(np.array(next_goal_, copy = False))
			action.append(np.array(action_, copy = False))
			reward.append(np.array(reward_, copy = False))
			done.append(np.array(done_, copy = False))
		return np.array(obs).reshape(batch_size, -1), \
			   np.array(local_goal), \
			   np.array(next_obs).reshape(batch_size, -1), \
			   np.array(next_goal), \
			   np.array(action).reshape(batch_size, -1), \
			   np.array(reward).reshape(batch_size, -1), \
			   np.array(done).reshape(batch_size, -1)