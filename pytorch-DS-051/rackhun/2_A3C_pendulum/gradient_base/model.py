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


class Global_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound):
        super(Global_Actor, self).__init__()

        self.state_dim = state_dim
        self.network = Actor_Net(state_dim, action_dim, action_bound)

    def predict(self, state):
        self.network.eval()
        with torch.no_grad():
            mu_a, _ = self.network.predict(state.view(1, self.state_dim))
        return mu_a

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.network.state_dict(),
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.network.load_state_dict(torch.load(os.path.join(path, name))['model'])


class Worker_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound, lr_rate, entropy_beta, global_actor):
        super(Worker_Actor, self).__init__()

        self.state_dim = state_dim
        self.std_bound = [1e-2, 1.]
        self.entropy_beta = entropy_beta

        self.global_actor = global_actor
        self.network = Actor_Net(state_dim, action_dim, action_bound)
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=lr_rate)
  

    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, min=self.std_bound[0], max=self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5*(action-mu)**2/var - 0.5*torch.log(var*2*2*torch.asin(torch.tensor(1.)))
        entropy = 0.5*(torch.log(2*2*torch.asin(torch.tensor(1.))*std**2)+1.)
        return torch.sum(log_policy_pdf, dim=1, keepdim=True), torch.sum(entropy, dim=1, keepdim=True)


    def get_action(self, state):
        self.network.eval()
        with torch.no_grad():
            mu_a, std_a = self.network(state.view(1, self.state_dim))
            mu_a, std_a = mu_a[0], std_a[0]
            std_a = torch.clamp(std_a, self.std_bound[0], self.std_bound[1])
            action = torch.normal(mu_a, std_a)
        return action


    def update(self, states, actions, advantages):
        self.network.train()
        mu_a, std_a = self.network(states)
        log_policy_pdf, entropy = self.log_pdf(mu_a, std_a, actions)
        loss = torch.sum(-log_policy_pdf*advantages.detach() - self.entropy_beta*entropy)

        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_actor.parameters(), self.network.parameters()):
            global_param._grad = local_param.grad
        torch_utils.clip_grad_norm_(self.global_actor.parameters(), 40.0)
        self.optimizer.step()


class Global_Critic(nn.Module):

    def __init__(self, state_dim):
        super(Global_Critic, self).__init__()

        self.network = Critic_Net(state_dim, 1)

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.network.state_dict(),
        }
        torch.save(state, os.path.join(path, name))

    def load(self, path, name):
        self.network.load_state_dict(torch.load(os.path.join(path, name))['model'])


class Worker_Critic(nn.Module):

    def __init__(self, state_dim, lr_rate, global_critic):
        super(Worker_Critic, self).__init__()

        self.global_critic = global_critic
        self.network = Critic_Net(state_dim, 1)
        self.optimizer = optim.Adam(self.global_critic.parameters(), lr=lr_rate)

    def get_value(self, states):
        self.network.eval()
        values = self.network(states)
        return values
    
    def update(self, states, td_targets):
        self.network.train()
        values = self.network(states)
        loss = F.mse_loss(values, td_targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_critic.parameters(), self.network.parameters()):
            global_param._grad = local_param.grad
        torch_utils.clip_grad_norm_(self.global_critic.parameters(), 40.0)
        self.optimizer.step()


 
