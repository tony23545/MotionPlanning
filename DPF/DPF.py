import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PFDataset import PFDataset
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DPF():
	def __init__(self, particle_dim = 3, action_dim = 2, observation_dim = 180, particle_num = 16, env = None):
		self.particle_dim = particle_dim
		self.state_dim = 4 # augmented state dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.particle_num = 16

		self.learning_rate = 0.0001
		self.propose_ratio = 0.0

		self.state_scale = torch.FloatTensor([1788.0, 1240.0, 1.0]).to(device)# not scaling angle, use sin and cos
		self.action_scale = torch.FloatTensor([20.0, 1.0]).to(device)
		self.observation_scale = 4.0
		self.delta_scale = torch.FloatTensor([2.0, 2.0, 20/7.5*np.tan(1.0)*0.1]).to(device)

		self.env = env

		self.build_model()

	def build_model(self):
		# observation model
		self.encoder = nn.Sequential(nn.Linear(self.observation_dim, 256),
									 nn.PReLU(),
									 nn.Linear(256, 128),
									 nn.PReLU(),
									 nn.Linear(128, 64)).to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = self.learning_rate)

		self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, 32),
										   nn.PReLU(),
										   nn.Linear(32, 64)).to(device)
		self.state_encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr = self.learning_rate)

		# observation likelihood estimator that maps states and observation encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(64+64, 128),
												nn.PReLU(),
												nn.Linear(128, 128),
												nn.PReLU(),
												nn.Linear(128, 64),
												nn.PReLU(),
												nn.Linear(64, 1),
												nn.Sigmoid()).to(device)
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = self.learning_rate)

		# particle proposer that maps encodings to particles
		self.particle_proposer = nn.Sequential(nn.Linear(64, 128),
											   nn.PReLU(),
											   nn.Linear(128, 128),
											   nn.PReLU(),
											   nn.Linear(128, 128),
											   nn.PReLU(),
											   nn.Linear(128, 64),
											   nn.PReLU(),
											   nn.Dropout(p=0.3),
											   nn.Linear(64, self.state_dim)).to(device)
		self.particle_proposer_optimizer = torch.optim.Adam(self.particle_proposer.parameters(), lr = self.learning_rate)

		# motion noise generator used for motion sampling 
		self.mo_noise_generator = nn.Sequential(nn.Linear(self.action_dim+1, 32),
												nn.PReLU(),
												nn.Linear(32, 32),
												nn.PReLU(),
												nn.Linear(32, self.action_dim),
												).to(device)
		self.mo_noise_generator_optimizer = torch.optim.Adam(self.mo_noise_generator.parameters(), lr = self.learning_rate)

		# transition_model maps augmented state and action to next state
		self.dynamic_model = nn.Sequential(nn.Linear(self.state_dim + self.action_dim, 64),
										   nn.PReLU(),
										   nn.Linear(64, 128),
										   nn.PReLU(),
										   nn.Linear(128, 64),
										   nn.PReLU(),
										   nn.Linear(64, self.particle_dim),
										   nn.Tanh()).to(device)
		self.dynamic_model_optimizer = torch.optim.Adam(self.dynamic_model.parameters(), lr = self.learning_rate)

	def transform_particles_as_input(self, particles):
		inputs = torch.cat((particles[..., :2],
							torch.sin(particles[..., 2:]),
							torch.cos(particles[..., 2:])), axis = -1)
		return inputs

	def measurement_update(self, encoding, particles):
		'''
		Compute the likelihood of the encoded observation for each particle.
		'''
		particle_input = self.transform_particles_as_input(particles)
		encoding_input = encoding[:, None, :].repeat((1, particle_input.shape[1], 1))
		state_encoding = self.state_encoder(particle_input)
		inputs = torch.cat((state_encoding, encoding_input), axis = -1)
		obs_likelihood = self.obs_like_estimator(inputs)
		return obs_likelihood 

	def motion_update(self, particles, action, training = False):
		action = action[:, None, :]
		action_input = action.repeat((1, particles.shape[1], 1))
		random_input = torch.rand(action_input.shape)[..., :1].to(device)
		action_random = torch.cat((action_input, random_input), axis = -1)
		
		# estimate action noise
		delta = self.mo_noise_generator(action_random)
		delta -= delta.mean(axis = 1, keepdim=True)
		
		noisy_actions = action + delta
		inputs = torch.cat((self.transform_particles_as_input(particles), noisy_actions), axis = -1)
		#inputs = self.transform_particles_as_input(torch.cat((particles, noisy_actions), axis = -1))
		# estimate delta and apply to current state
		state_delta = self.dynamic_model(inputs)
		return state_delta

	def propose_particles(self, encoding, num_particles):
		duplicated_encoding = encoding[:, None, :].repeat((1, num_particles, 1)).to(device)
		proposed_particles = self.particle_proposer(duplicated_encoding).cpu()
		proposed_particles[..., :2] = torch.sigmoid(proposed_particles[..., :2])
		proposed_particles[..., 2:] = torch.tanh(proposed_particles[..., 2:])
		proposed_particles = torch.cat((proposed_particles[..., :2],
										torch.atan2(proposed_particles[..., 2:3], proposed_particles[..., 3:4])), axis = -1)
		proposed_particles[..., 2] = proposed_particles[..., 2] + np.pi
		return proposed_particles

	def resample(self, particles, particle_probs, alpha, num_resampled):
		'''
		particle_probs in log space, unnormalized
		'''
		assert 0.0 < alpha <= 1.0
		batch_size = particles.shape[0]		

		# normalize
		particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
		uniform_probs = torch.ones((batch_size, self.particle_num)).to(device) / self.particle_num

		# bulid up sampling distribution q(s)
		if alpha < 1.0:
			# soft resampling
			q_probs = torch.stack((particle_probs*alpha, uniform_probs*(1.0-alpha)), dim = -1).to(device)
			q_probs = q_probs.sum(dim = -1)
			q_probs = q_probs / q_probs.sum(dim = -1, keepdim = True)
			particle_probs = particle_probs / q_probs
		else:
			# hard resampling
			q_probs = particle_probs
			particle_probs = uniform_probs

		# sample particle indices according to q(s)

		basic_markers = torch.linspace(0.0, (num_resampled-1.0)/num_resampled, num_resampled)
		random_offset = torch.FloatTensor(batch_size).uniform_(0.0, 1.0/num_resampled)
		markers = random_offset[:, None] + basic_markers[None, :] # shape: batch_size * num_resampled
		cum_probs = torch.cumsum(q_probs, axis = 1)
		markers = markers.to(device)
		marker_matching = markers[:, :, None] > cum_probs[:, None, :]
		samples = marker_matching.sum(axis = 2).int()

		idx = samples + self.particle_num*torch.arange(batch_size)[:, None].repeat((1, num_resampled)).to(device)
		particles_resampled = particles.view((batch_size * self.particle_num, -1))[idx, :]
		particle_probs_resampled = particle_probs.view((batch_size * self.particle_num, ))[idx]
		particle_probs_resampled = particle_probs_resampled / particle_probs_resampled.sum(dim = -1, keepdim = True)


		return particles_resampled, particle_probs_resampled

	def propose_batch(self, observation, num_proposed = 100):
		with torch.no_grad():
			encoding = self.encoder(torch.FloatTensor(observation / 4.0).to(device))
			particles_proposed = []
			scale = self.state_scale.cpu()
			while True:
				prop = self.propose_particles(encoding, num_proposed*2)
				for k in range(prop.shape[0]):
					if self.env.state_validity_checker((prop[:, k] * scale).numpy().T):
						particles_proposed.append(prop[0, k])
					if len(particles_proposed) >= num_proposed:
						break
				if len(particles_proposed) >= num_proposed:
					break
			particles_proposed = torch.stack(particles_proposed)[None, ...]
			return particles_proposed

	def loop(self, particles, particle_probs, actions, observation, training = False):
		encoding = self.encoder(observation)

		# motion update
		deltas = self.motion_update(particles, actions)
		particles = particles + deltas * self.delta_scale / self.state_scale

		# manually clear invalid particles 
		temp = (particles[0] * torch.FloatTensor([[1780, 1240, 1]])).numpy().T
		for i in range(particles.shape[1]):
			if not self.env.state_validity_checker(temp[:, i:(i+1)]):
				particle_probs[0, i] = 0.0

		# observation update
		likelihood = (self.measurement_update(encoding, particles).squeeze()+1e-16)
		#print(likelihood.max())
		particle_probs = particle_probs * likelihood # unnormalized

		if likelihood.max() < 0.9:
			propose_ratio = 0.2
		else:
			propose_ratio = 0.00

		num_proposed = int(self.particle_num * propose_ratio)
		num_resampled = self.particle_num - num_proposed

		# resample
		alpha = 0.8
		particles_resampled, particle_probs_resampled = self.resample(particles, particle_probs, alpha, num_resampled)

		# propose
		if num_proposed > 0:
			particles_proposed = []
			scale = self.state_scale.cpu()
			while True:
				prop = self.propose_particles(encoding, num_proposed*2)
				for k in range(prop.shape[0]):
					if self.env.state_validity_checker((prop[:, k] * scale).numpy().T):
						particles_proposed.append(prop[0, k])
					if len(particles_proposed) >= num_proposed:
						break
				if len(particles_proposed) >= num_proposed:
					break
			particles_proposed = torch.stack(particles_proposed)[None, ...].to(device)

			particle_probs_proposed = torch.ones([particles_proposed.shape[0], particles_proposed.shape[1]]) / particles_proposed.shape[1]
			particle_probs_proposed = particle_probs_proposed.to(device)

			# combine
			particles = torch.cat((particles_resampled, particles_proposed), axis = 1)
			particle_probs = torch.cat((particle_probs_resampled, particle_probs_proposed), axis = -1)
			particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
			return particles, particle_probs
		else:
			return particles_resampled, particle_probs_resampled

	def train(self, loader, max_iter=1000):
		# no motion model here...
		# train dynamic model
		mseLoss = nn.MSELoss()
		#TODO can train dynamic and measurement at the same time...
		print("training motion model...")
		#self.load()
		for it in range(max_iter):
			total_loss = []
			for _, (states, actions, measurements, next_states, deltas) in enumerate(loader):
				states = states.float().to(device)
				actions = actions.float().to(device)
				next_states = next_states.float().to(device)
				deltas = deltas.float().to(device)
				states = states[:, None, :]

				delta_pred = self.motion_update(states, actions, training = True)
				dynamic_loss = mseLoss(delta_pred.squeeze(), deltas)

				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				dynamic_loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				total_loss.append(dynamic_loss.detach().cpu().numpy())
			print("epoch: %d, loss: %2.6f" % (it, np.mean(total_loss)))
		
		# train measurement model
		print("training measurement model...")
		for it in range(max_iter):
			total_loss = []
			for _, (states, actions, measurements, next_states, deltas) in enumerate(loader):
				batch_size = states.shape[0]

				state_repeat = next_states[None, ...].repeat(batch_size, 1, 1)
				state_repeat = state_repeat.float().to(device)
				measurements = measurements.float().to(device)
				encoding = self.encoder(measurements)
				measurement_model_out = self.measurement_update(encoding, state_repeat).squeeze().cpu()

				# measure_loss = -torch.mul(torch.eye(batch_size), torch.log(measurement_model_out + 1e-16))/batch_size -  \
				# 				torch.mul(1.0-torch.eye(batch_size), torch.log(1.0-measurement_model_out + 1e-16))/(batch_size**2-batch_size)
				# measure_loss_mean = measure_loss.sum()

				# measure_loss = -torch.mul(torch.eye(batch_size), torch.log(measurement_model_out + 1e-16))/batch_size -  \
				# 				torch.mul(1.0-torch.eye(batch_size), torch.log(1.0-measurement_model_out + 1e-16))/(batch_size**2-batch_size)
				measure_loss = -torch.mul(torch.mul(torch.eye(batch_size), torch.pow(1.0-measurement_model_out, 2.0)), torch.log(measurement_model_out + 1e-16))/batch_size - \
								torch.mul(torch.mul(1.0-torch.eye(batch_size), torch.pow(measurement_model_out, 2.0)), torch.log(1.0-measurement_model_out + 1e-16))/(batch_size**2-batch_size)

				measure_loss_mean = measure_loss.sum()

				self.encoder_optimizer.zero_grad()
				self.state_encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				measure_loss_mean.backward()
				self.encoder_optimizer.step()
				self.state_encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				total_loss.append(measure_loss_mean.detach().cpu().numpy())
			print("epoch: %d, loss: %2.6f" % (it, np.mean(total_loss)))

		# train particle proposer
		print("training proposer...")
		for it in range(max_iter):
			total_loss = []
			for _, (states, actions, measurements, next_states, deltas) in enumerate(loader):
				
				measurements = measurements.float().to(device)
				encoding = self.encoder(measurements).detach()

				proposed_particles = self.propose_particles(encoding, self.particle_num)

				state_repeat = next_states[:, None, :].repeat((1, self.particle_num, 1))
				state_repeat[..., 2] = state_repeat[..., 2]
				std = 0.2
				sq_distance = (proposed_particles - state_repeat).pow(2).sum(axis = -1)
				activations = 1.0 / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
				proposer_loss = -torch.log(1e-16 + activations.mean(axis = -1)).mean()
				self.particle_proposer_optimizer.zero_grad()
				proposer_loss.backward()
				self.particle_proposer_optimizer.step()
				total_loss.append(proposer_loss.detach().cpu().numpy())
			print("epoch: %d, loss: %2.6f" % (it, np.mean(total_loss)))

		# # end to end training
		# print("end to end training...")
		# for it in range(max_iter):
		# 	total_loss = []
		# 	sq_loss = []
		# 	for _, (states, actions, measurements, next_states, deltas) in enumerate(loader):
		# 		batch_size = states.shape[0]
		# 		# import IPython
		# 		# IPython.embed()
		# 		particles = states[:, None, :].repeat((1, self.particle_num, 1))
		# 		particle_probs = torch.ones((batch_size, self.particle_num)) / self.particle_num

		# 		particles = particles.float().to(device)
		# 		particle_probs = particle_probs.float().to(device)
		# 		actions = actions.float().to(device)
		# 		measurements = measurements.float().to(device)
		# 		next_particles, next_particle_probs = self.loop(particles, particle_probs, actions, measurements, training = True)
				
		# 		std = 0.2
				
		# 		next_state_repeat = next_states[:, None, :].repeat((1, self.particle_num, 1))
		# 		next_state_repeat[..., 2] = next_state_repeat[..., 2]
		# 		sq_distance = (next_particles - next_state_repeat).pow(2).sum(axis = -1)
		# 		activations = next_particle_probs / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
		# 		e2e_loss = -torch.log(1e-16 + activations.sum(axis = -1)).mean()
				
		# 		mseloss = (next_particle_probs * sq_distance).sum(axis=-1).mean()
		# 		#mean_next_state = self.particles_to_state(next_particles, next_particle_probs)

		# 		# update all parameters
		# 		self.mo_noise_generator_optimizer.zero_grad()
		# 		self.dynamic_model_optimizer.zero_grad()
		# 		self.encoder_optimizer.zero_grad()
		# 		self.obs_like_estimator_optimizer.zero_grad()
		# 		e2e_loss.backward()
		# 		self.mo_noise_generator_optimizer.step()
		# 		self.dynamic_model_optimizer.step()
		# 		self.encoder_optimizer.step()
		# 		self.obs_like_estimator_optimizer.step()
		# 		total_loss.append(e2e_loss.cpu().detach().numpy())
		# 		sq_loss.append(mseloss.cpu().detach().numpy())
		# 	print("epoch: %d, loss: %2.4f, %2.4f"  % (it, np.mean(total_loss), np.mean(sq_loss)))

	def initial_particles(self, particles):
		self.particles = torch.FloatTensor(particles).to(device)
		self.particle_num = self.particles.shape[1]
		self.particle_probs = torch.ones((1, self.particle_num)) / self.particle_num
		self.particle_probs = self.particle_probs.to(device)

	def update(self, action, obs):
		action = torch.FloatTensor(action)[None, :].to(device) / self.action_scale
		obs = torch.FloatTensor(obs) / self.observation_scale
		action = action.to(device)
		obs = obs.to(device)
		with torch.no_grad():
			new_particles, new_particle_probs = self.loop(self.particles, self.particle_probs, action, obs)
		self.particles = new_particles
		self.particle_probs = new_particle_probs

		return (new_particles * self.state_scale).cpu(), self.particle_probs.cpu()

	def load(self, file_name = None):
		try:
			if file_name is None:
				file_name = "weights/DPFfocal.pt"
			checkpoint = torch.load(file_name, map_location=torch.device(device))
			self.encoder.load_state_dict(checkpoint["encoder"])
			self.state_encoder.load_state_dict(checkpoint["state_encoder"])
			self.obs_like_estimator.load_state_dict(checkpoint["obs_like_estimator"])
			self.particle_proposer.load_state_dict(checkpoint["particle_proposer"])
			self.mo_noise_generator.load_state_dict(checkpoint["mo_noise_generator"])
			self.dynamic_model.load_state_dict(checkpoint["dynamic_model"])
			print("load model from " + file_name)
		except:
			print("fail to load model! ")

	def save(self):
		file_name = "weights/DPFfocal.pt"
		torch.save({"encoder" : self.encoder.state_dict(),
					"state_encoder" : self.state_encoder.state_dict(),
					"obs_like_estimator" : self.obs_like_estimator.state_dict(),
					"particle_proposer" : self.particle_proposer.state_dict(),
					"mo_noise_generator" : self.mo_noise_generator.state_dict(),
					"dynamic_model" : self.dynamic_model.state_dict()}, file_name)
		print("save model to " + file_name)

if __name__ == "__main__":
	dataset = PFDataset()
	loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)

	dpf = DPF()
	dpf.load()
	for i in range(1):
		dpf.train(loader, 50)
		dpf.save()

