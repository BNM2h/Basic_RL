import os
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils as torch_utils


class Actor_Net(nn.Module):

    def __init__(self, n_in, n_out, action_bound):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_out  = nn.Linear(16, n_out)

        self.action_bound = action_bound
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
       
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))

        out  = F.tanh(self.fc_out(h3))
        return out*self.action_bound


class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, action_bound, lr_rate, tau):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.train_actor = Actor_Net(state_dim, action_dim, action_bound)
        self.target_actor= Actor_Net(state_dim, action_dim, action_bound)
        self.optimizer = optim.Adam(self.train_actor.parameters(), lr=lr_rate)
        self.tau = tau

    def predict(self, state):
        self.train_actor.eval()
        return self.train_actor(state)

    def target_predict(self, state):
        self.target_actor.eval()
        return self.target_actor(state)

    def train(self, policy_loss):
        self.train_actor.train()
        loss = -torch.mean(policy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_param, train_param in zip(self.target_actor.parameters(), self.train_actor.parameters()):
            target_param.data.copy_( self.tau*train_param.data + (1-self.tau)*target_param.data)


    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.target_actor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.target_actor.load_state_dict(torch.load(os.path.join(path, name))['model'])


class Critic_Net(nn.Module):

    def __init__(self, n_state, n_action, n_out):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc2 = nn.Linear(64, 32)

        self.fca = nn.Linear(n_action, 32)
        self.fc3 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16, n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        h1 = F.relu(self.fc1(state))
        h2 = self.fc2(h1)

        a1 = self.fca(action)
        h2 = torch.cat([h2, a1], dim=1)
        h3 = F.relu(self.fc3(h2))
        out = self.fc_out(h3)
        return out


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, lr_rate, tau):
        super(Critic, self).__init__()

        self.train_critic  = Critic_Net(state_dim, action_dim, 1)
        self.target_critic = Critic_Net(state_dim, action_dim, 1)
        self.optimizer = optim.Adam(self.train_critic.parameters(), lr=lr_rate)
        self.tau = tau

    def predict(self, states, actions):
        self.train_critic.eval()
        return self.train_critic(states, actions)

    def target_predict(self, states, actions):
        self.target_critic.eval()
        return self.target_critic(states, actions)

    def train(self, states, actions, targets):
        self.train_critic.train()
        values = self.train_critic(states, actions)
        loss = F.mse_loss(values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_param, train_param in zip(self.target_critic.parameters(), self.train_critic.parameters()):
            target_param.data.copy_( self.tau*train_param.data + (1-self.tau)*target_param.data)

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.target_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.target_critic.load_state_dict(torch.load(os.path.join(path, name))['model'])
