import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PFDataset import PFDataset
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DPF():
	def __init__(self, particle_dim = 3, action_dim = 2, observation_dim = 180, particle_num = 16):
		self.particle_dim = particle_dim
		self.state_dim = 4 # augmented state dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.particle_num = 16

		self.learning_rate = 0.0001
		self.propose_ratio = 0.0

		self.state_scale = np.array([1788, 1240, 1.0])# not scaling angle, use sin and cos
		self.action_scale = np.array([20, 1.0])
		self.observation_scale = 4.0
		self.delta_pos_scale = np.array([1788, 1240, 2.0]) / 2.0
		self.delta_angle_scale = 1. / (20/7.5*np.tan(1.0)*0.1)
		self.env = None

		self.build_model()

	def build_model(self):
		# observation model
		self.encoder = nn.Sequential(nn.Linear(self.observation_dim, 256),
									 nn.PReLU(),
									 nn.Linear(256, 128),
									 nn.PReLU(),
									 nn.Linear(128, 64)).to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = self.learning_rate)


		# observation likelihood estimator that maps states and observation encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(self.state_dim+64, 128),
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
											   nn.Dropout(p=0.2),
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
		inputs = torch.cat((particle_input, encoding_input), axis = -1)
		obs_likelihood = self.obs_like_estimator(inputs)
		return obs_likelihood 

	def propose_particles(self, encoding, num_particles):
		duplicated_encoding = encoding[:, None, :].repeat((1, num_particles, 1)).to(device)
		proposed_particles = self.particle_proposer(duplicated_encoding).cpu()
		proposed_particles = torch.cat((proposed_particles[..., :2],
										torch.atan2(proposed_particles[..., 2:3], proposed_particles[..., 3:4])), axis = -1)
		proposed_particles[..., 2] = proposed_particles[..., 2] + np.pi
		return proposed_particles

	def motion_update(self, particles, action, training = False):
		action = action[:, None, :]
		action_input = action.repeat((1, particles.shape[1], 1))
		random_input = torch.rand(action_input.shape)[..., :1]
		action_random = torch.cat((action_input, random_input), axis = -1).to(device)
		
		# estimate action noise
		delta = self.mo_noise_generator(action_random)
		delta -= delta.mean(axis = 1, keepdim=True)
		
		noisy_actions = action + delta
		inputs = torch.cat((self.transform_particles_as_input(particles), noisy_actions), axis = -1)
		#inputs = self.transform_particles_as_input(torch.cat((particles, noisy_actions), axis = -1))
		# estimate delta and apply to current state
		state_delta = self.dynamic_model(inputs)
		return state_delta

	def permute_batch(self, x, samples):
		# get shape
		batch_size = x.shape[0]
		num_particles = x.shape[1]
		sample_size = samples.shape[1]
		# compute 1D indices into the 2D array
		idx = samples + num_particles*torch.arange(batch_size)[:, None].repeat((1, sample_size))
		result = x.view(batch_size*num_particles, -1)[idx, :]
		return result

	def loop(self, particles, particle_probs_, actions, imgs, training = False):
		encoding = self.encoder(imgs)
		num_proposed = int(self.particle_num * self.propose_ratio)
		num_resampled = self.particle_num - num_proposed
		batch_size = encoding.shape[0]

		if self.propose_ratio == 0:
			#standard_particles = particles
			#standard_particles_probs = particle_probs_
			# motion update
			
			standard_particles = self.motion_update(particles, actions, training) + particles

			# measurement update
			likelihood = (self.measurement_update(encoding, standard_particles).squeeze()+1e-16)
			standard_particles_probs = likelihood * particle_probs_
		elif self.propose_ratio < 1.0:
			# resampling
			basic_markers = torch.linspace(0.0, (num_resampled-1.0)/num_resampled, num_resampled)
			random_offset = torch.FloatTensor(batch_size).uniform_(0.0, 1.0/num_resampled)
			markers = random_offset[:, None] + basic_markers[None, :] # shape: batch_size * num_resampled
			cum_probs = torch.cumsum(particle_probs_, axis = 1)
			markers = markers.to(device)
			marker_matching = markers[:, :, None] > cum_probs[:, None, :] # shape: batch_size * num_resampled * num_particles
			#samples = marker_matching.int().argmax(axis = 2).int()
			samples = marker_matching.sum(axis = 2).int()
			#print(samples)
			standard_particles = self.permute_batch(particles, samples)
			standard_particles_probs = torch.ones((batch_size, num_resampled)).to(device)

			# motion update
			standard_particles = self.motion_update(standard_particles, actions, training) + standard_particles

			# measurement update
			standard_particles_probs *= (self.measurement_update(encoding, standard_particles).squeeze()+1e-16)

		if self.propose_ratio > 0.0:
			# propose new particles

			proposed_particles = self.propose_particles(encoding.detach(), num_proposed)
			proposed_particles_probs = torch.ones((batch_size, num_proposed)).to(device)

		# normalize and combine particles
		if self.propose_ratio == 1.0:
			particles = propose_particles
			particle_probs = proposed_particles_probs

		elif self.propose_ratio == 0.0:
			particles = standard_particles
			particle_probs = standard_particles_probs

		else:
			standard_particles_probs *= (1.0 * num_resampled / self.particle_num / standard_particles_probs.sum(axis = 1, keepdim=True))
			proposed_particles_probs *= (1.0 * num_proposed / self.particle_num / proposed_particles_probs.sum(axis = 1, keepdim=True))
			particles = torch.cat((standard_particles, proposed_particles), axis = 1)
			particle_probs = torch.cat((standard_particles_probs, proposed_particles_probs), axis = 1)

		# normalize probabilities
		particle_probs /= (particle_probs.sum(axis = 1, keepdim = True))
	
		return particles, particle_probs

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
				deltas = deltas.float().to(device)
				states = states[:, None, :]

				delta_pred = self.motion_update(states, actions, training = True)
				dynamic_loss = mseLoss(delta_pred.squeeze(), deltas)

				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				dynamic_loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				total_loss.append(dynamic_loss.detach().numpy())
				self.dynamic_model_optimizer.step()
				total_loss.append(dynamic_loss.detach().numpy())
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
				measurement_model_out = self.measurement_update(encoding, state_repeat).squeeze()

				measure_loss = -torch.mul(torch.eye(batch_size), torch.log(measurement_model_out + 1e-16))/batch_size -  \
								torch.mul(1.0-torch.eye(batch_size), torch.log(1.0-measurement_model_out + 1e-16))/(batch_size**2-batch_size)
				measure_loss_mean = measure_loss.sum()

				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				measure_loss_mean.backward()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				total_loss.append(measure_loss_mean.detach().numpy())
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
				total_loss.append(proposer_loss.detach().numpy())
			print("epoch: %d, loss: %2.6f" % (it, np.mean(total_loss)))
		self.save()

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
		# 		self.particle_proposer_optimizer.zero_grad()
		# 		e2e_loss.backward()
		# 		self.mo_noise_generator_optimizer.step()
		# 		self.dynamic_model_optimizer.step()
		# 		self.encoder_optimizer.step()
		# 		self.obs_like_estimator_optimizer.step()
		# 		self.particle_proposer_optimizer.step()
		# 		total_loss.append(e2e_loss.cpu().detach().numpy())
		# 		sq_loss.append(mseloss.cpu().detach().numpy())
		# 	print("epoch: %d, loss: %2.4f, %2.4f"  % (it, np.mean(total_loss), np.mean(sq_loss)))

	def load(self):
		try:
			if not os.path.exists("../weights/"):
				os.mkdir("../weights/")
			file_name = "../weights/DPF1.pt"
			checkpoint = torch.load(file_name)
			self.encoder.load_state_dict(checkpoint["encoder"])
			self.obs_like_estimator.load_state_dict(checkpoint["obs_like_estimator"])
			self.particle_proposer.load_state_dict(checkpoint["particle_proposer"])
			self.mo_noise_generator.load_state_dict(checkpoint["mo_noise_generator"])
			self.dynamic_model.load_state_dict(checkpoint["dynamic_model"])
			print("load model from " + file_name)
		except:
			print("fail to load model!")

	def save(self):
		if not os.path.exists("../weights/"):
			os.mkdir("../weights/")
		file_name = "../weights/DPF1.pt"
		torch.save({"encoder" : self.encoder.state_dict(),
					"obs_like_estimator" : self.obs_like_estimator.state_dict(),
					"particle_proposer" : self.particle_proposer.state_dict(),
					"mo_noise_generator" : self.mo_noise_generator.state_dict(),
					"dynamic_model" : self.dynamic_model.state_dict()}, file_name)
		print("save model to " + file_name)

if __name__ == "__main__":
	dataset = PFDataset()
	loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)

	dpf = DPF()
	for i in range(1):
		dpf.train(loader, 50)

