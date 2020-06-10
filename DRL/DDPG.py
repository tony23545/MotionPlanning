import argparse
from itertools import count

import os, sys, random
import numpy as np
import _pickle as pickle 

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.models import QNetwork, DeterministicPolicy
from utils.ReplayBuffer import ReplayBuffer

import sys
sys.path.append('../Env')
from CarEnvironment import CarEnvironment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPG():
	def __init__(self, args, env = None):
		self.args = args
		# actor
		self.actor = DeterministicPolicy(128)
		self.actor_target = DeterministicPolicy(128)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.lr)
		# critics
		self.critic = QNetwork(128).to(device)
		self.critic_target = QNetwork(128).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), self.args.lr)

		self.replay_buffer = ReplayBuffer(self.args.capacity)
		self.num_critic_update_iteration = 0
		self.num_actor_update_iteration = 0
		self.num_training = 0
		self.global_steps = 0

		self.action_scale = torch.FloatTensor([[20, 1]])
		self.env = env
		#self.load()

	def update(self):
		for it in range(self.args.update_iteration):
			# sample from replay buffer
			obs, local_goal, next_obs, next_goal, action, reward, done = self.replay_buffer.sample(self.args.batch_size)
			obs = torch.FloatTensor(obs).to(device)
			local_goal = torch.FloatTensor(local_goal).to(device)
			next_obs = torch.FloatTensor(next_obs).to(device)
			next_goal = torch.FloatTensor(next_goal).to(device)
			action = torch.FloatTensor(action).to(device)
			reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(done).to(device)

			# computer the target Q value
			next_action, _ = self.actor_target.sample(next_obs, next_goal)
			target_Q = self.critic_target(next_obs, next_goal, next_action / self.action_scale)
			target_Q = reward + ((1-done) * self.args.gamma * target_Q).detach()

			# get current Q estimate
			current_Q = self.critic(obs, local_goal, action)

			# compute cirtic loss and update
			critic_loss = F.mse_loss(current_Q, target_Q)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# computer actor loss
			actor_action, _ = self.actor.sample(obs, local_goal)
			actor_loss = -self.critic(obs, local_goal, actor_action / self.action_scale).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# update target model 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

			self.num_actor_update_iteration += 1
			self.num_critic_update_iteration += 1

	def train(self):
		for i in range(self.args.max_episode):
			obs, local_goal = self.env.reset()
			ep_r = 0

			for t in count():
				action, _ = self.actor.sample(torch.FloatTensor(obs).to(device), torch.FloatTensor(local_goal).to(device))
				action = action.cpu().detach().numpy()[0]

				next_obs, next_goal, done, reward = self.env.step(action)
				self.global_steps += 1
				ep_r += reward
				self.replay_buffer.push((obs / 4.0, local_goal / 20., next_obs / 4.0, next_goal / 20., action / np.array([20, 1]), reward, np.float(done)))
				obs = next_obs
				local_goal = next_goal

				if done or t > self.args.max_length_trajectory:
					if i % self.args.print_log == 0:
						print("Ep_i \t {}, the ep_r is \t{:0.2f}, the step is \t{}, global_steps is {}".format(i, ep_r, t, self.global_steps))
						self.evaluate(10, False)
					break

			if len(self.replay_buffer.storage) >= self.args.capacity * 0.2:
				self.update()

		self.save()

	def evaluate(self, number = 1, render = True):
		rewards = []
		for _ in range(number):
			total_rews = 0
			time_step = 0
			done = False
			obs, local_goal = self.env.reset()
			while not done:
				action = self.predict(obs / 4., local_goal / 20.)
				# with torch.no_grad():
				# 	# use the mean action
				# 	_, action = self.actor.sample(torch.FloatTensor(obs).to(device) / 4., torch.FloatTensor(local_goal).to(device) / 20)
				# 	action = action.cpu().detach().numpy()[0]

				obs, local_goal, done, reward = self.env.step(action)
				
				if render:
					self.env.render()
				total_rews += reward
				time_step += 1
				if time_step > self.args.max_length_trajectory:
					break
				#print(str(action) + "  " + str(local_goal))
				if done:
					break

			rewards.append(total_rews)
		rewards = np.array(rewards)
		print("mean reward {}, max reward {}, min reward {}".format(rewards.mean(), rewards.max(), rewards.min()))

	def predict(self, obs, local_goal):
		with torch.no_grad():
			_, action = self.actor.sample(torch.FloatTensor(obs).to(device), torch.FloatTensor(local_goal).to(device))
		action = action.cpu().detach().numpy()[0]
		return action

	def load(self, episode = None):
		file_name = "weights/DDPG.pt"
		checkpoint = torch.load(file_name)
		self.actor.load_state_dict(checkpoint['actor'])
		self.actor_target.load_state_dict(checkpoint['actor_target'])
		self.critic.load_state_dict(checkpoint['critic'])
		self.critic.load_state_dict(checkpoint['critic_target'])
		print("successfully load model from " + file_name)

	def save(self, episode = None):
		file_name = "weights/DDPG.pt"
		torch.save({'actor' : self.actor.state_dict(),
					'critic' : self.critic.state_dict(),
					'actor_target' : self.actor_target.state_dict(),
					'critic_target' : self.critic_target.state_dict()}, file_name)
		print("save model to " + file_name)
