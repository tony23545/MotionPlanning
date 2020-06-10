import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class MLP(nn.Module):
    def __init__(self, layers, activation = None):
        super(MLP, self).__init__()
        modules = []
        l = len(layers)
        for i in range(l - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[l-2], layers[l-1]))
        if not activation == None:
            modules.append(activation)
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, hidden_dim = 128):
        super(QNetwork, self).__init__()
        # observation encoder
        self.obs_encoder = MLP([180, 128, 64])

        # goal, action encoder
        self.goal_action_encoder = MLP([4, 32, 64])

        # Q net
        self.QNet = MLP([64, hidden_dim, hidden_dim, hidden_dim, 1])

        self.apply(weights_init_)

    def forward(self, observation, goal, action):
        obs_encoding = self.obs_encoder(observation)
        ga = torch.cat([goal, action], 1)
        ga_encoding = self.goal_action_encoder(ga)
        qvalue = self.QNet(obs_encoding + ga_encoding)
        return qvalue

class DeterministicPolicy(nn.Module):
    def __init__(self, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        # observation encoder
        self.obs_encoder = MLP([180, 128, 64])

        # goal encoder
        self.goal_encoder = MLP([2, 32, 64])

        # policy
        self.policy = MLP([64, hidden_dim, hidden_dim, hidden_dim, 2], nn.Tanh())
        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor([[20, 1]])
        self.noise = torch.Tensor(2)

    def forward(self, observation, goal):
        obs_encoding = self.obs_encoder(observation)
        goal_encoding = self.goal_encoder(goal)
        action = self.policy(obs_encoding + goal_encoding) * self.action_scale
        return action

    def sample(self, observation, goal):
        mean = self.forward(observation, goal)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.1, 0.1) * self.action_scale
        action = mean + noise
        return action, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        #self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
