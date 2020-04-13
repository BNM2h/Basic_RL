import os
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Actor_Net(nn.Module):

    def __init__(self, n_in, n_out, action_bound):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

        self.fc_mu  = nn.Linear(16, n_out)
        self.fc_std = nn.Linear(16, n_out)

        self.action_bound = action_bound

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))

        mu  = F.tanh(self.fc_mu(h3))
        std = F.softplus(self.fc_std(h3))
        return mu*self.action_bound, std


class Critic_Net(nn.Module):

    def __init__(self, n_in, n_out):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_value  = nn.Linear(16, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = self.fc_value(h3)
        return out


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound, lr_rate):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.std_bound = [1e-2, 1.]
        self.actor_network = Actor_Net(state_dim, action_dim, action_bound)
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=lr_rate)

    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5*(action-mu)**2/var - 0.5*torch.log(var*2*torch.asin(torch.tensor(1.)))
        return torch.sum(log_policy_pdf, dim=1, keepdim=True)

    def get_action(self, state):
        self.actor_network.eval()
        with torch.no_grad():
            mu_a, std_a = self.actor_network(state.view(1, self.state_dim))
            mu_a, std_a = mu_a[0], std_a[0]
            std_a = torch.clamp(std_a, self.std_bound[0], self.std_bound[1])
            action = torch.normal(mu_a, std_a)
        return action

    #def predict(self, state):
    #    mu_a, _ = self.actor_network(state.view(1, self.state_dim))
    #    return mu_a[0]

    def update(self, states, actions, advantages):
        self.actor_network.train()
        mu_a, std_a = self.actor_network(states)
        log_policy_pdf = self.log_pdf(mu_a, std_a, actions)
        loss = torch.sum(-log_policy_pdf*advantages)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.actor_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.actor_network.load_state_dict(torch.load(os.path.join(path, name))['model'])


class Critic(nn.Module):

    def __init__(self, state_dim, lr_rate):
        super(Critic, self).__init__()

        self.critic_network = Critic_Net(state_dim, 1)
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=lr_rate)

    def get_value(self, states):
        self.critic_network.eval()
        values = self.critic_network(states)
        return values
            
    def update(self, states, targets):
        self.critic_network.train()
        values = self.critic_network(states)
        loss = F.mse_loss(values, targets)
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.critic_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.critic_network.load_state_dict(torch.load(os.path.join(path, name))['model'])


class Agent:
  
    def __init__(self, state_dim, action_dim, action_bound, batch_size, lr_rate, gamma):

        self.gamma = gamma
        self.batch = batch_size
        self.actor_lr_rate = lr_rate[0]
        self.critic_lr_rate = lr_rate[1]
        self.action_bound = action_bound
        
        self.actor  = Actor(state_dim, action_dim, action_bound, lr_rate[0])
        self.critic = Critic(state_dim, lr_rate[1])

    def get_action(self, states):
        return self.actor.get_action(states)

    def critic_predict(self, states):
        return self.critic.get_value(states)

    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = reward
            advantage = y_k - v_value
        else:
            y_k = reward + self.gamma*next_v_value
            advantage = y_k - v_value
        return advantage, y_k

    def critic_train(self, states, td_targets):
        self.critic.update(states, td_targets)

    def actor_train(self, states, actions, advantages):
        self.actor.update(states, actions, advantages)
 
