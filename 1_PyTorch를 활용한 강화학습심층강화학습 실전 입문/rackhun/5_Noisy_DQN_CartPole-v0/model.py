import math
import random
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


class NoisyLinear(nn.Module):
    
    def __init__(self, in_features, out_features, std_init=0.4, is_training=True):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.is_training = is_training

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):

        if self.is_training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x



class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out, is_training=True):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.noisy_fc2 = NoisyLinear(n_mid, n_mid, is_training=is_training)
        # Dueling Network
        self.noisy_fc3_adv = NoisyLinear(n_mid, n_out, is_training=is_training)
        self.noisy_fc3_v   = NoisyLinear(n_mid, 1, is_training=is_training)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.noisy_fc2(h1))

        adv = self.noisy_fc3_adv(h2)   # No activation
        val = self.noisy_fc3_v(h2).expand(-1, adv.size(1))   # No activation

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3_adv.reset_noise()
        self.noisy_fc3_v.reset_noise()



class TDerrorMemory:

    def __init__(self, capacity, td_error_epsilon):
        self.capacity = capacity
        self.td_error_epsilon = td_error_epsilon
        self.memory = []
        self.index = 0

    def push(self, td_error):
        """ save td_error on memory """

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        self.index = (self.index+1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += self.td_error_epsilon*len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx]) + self.td_error_epsilon)
                idx += 1

            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors



class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        """ save transition(state, action, state_next, reward) on memory """

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index+1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Brain:

    def __init__(self, num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, td_epsilon, is_training, device):
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity)
        self.td_error_memory = TDerrorMemory(capacity, td_epsilon)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.main_q_network   = Net(num_states, hidden_size, self.num_actions, is_training)
        self.target_q_network = Net(num_states, hidden_size, self.num_actions, is_training)

        if self.device != torch.device('cpu'):
            self.main_q_network   = self.main_q_network.cuda(self.device)
            self.target_q_network = self.target_q_network.cuda(self.device)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr_rate)

    def replay(self, episode):

        #1. check memory
        if len(self.memory) < self.batch_size:
            return

        #2. mini batch 
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch(episode)

        #3. target Q
        self.expected_state_action_values = self.get_expected_state_action_values()

        #4. Update
        self.update_main_q_network()


    def decide_action(self, state, episode):

        if self.device != torch.device('cpu'):
            state = state.cuda(self.device)

        self.main_q_network.eval()
        with torch.no_grad():
            action = self.main_q_network(state).max(1)[1].view(1,1)

        if self.device == torch.device('cpu'):
            return action
        else:
            return action.cuda(self.device)


    def make_minibatch(self, episode):
        if episode < 30:
            transitions = self.memory.sample(self.batch_size)
        else:
            indexes = self.td_error_memory.get_prioritized_indexes(self.batch_size)
            transitions = [self.memory.memory[n] for n in indexes]

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        if self.device == torch.device('cpu'):
            return batch, state_batch, action_batch, reward_batch, non_final_next_states
        else:
            return batch, state_batch.cuda(self.device), action_batch.cuda(self.device), reward_batch.cuda(self.device), non_final_next_states.cuda(self.device)


    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        next_state_values = torch.zeros(self.batch_size)
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor)

        if self.device != torch.device('cpu'):
            next_state_values = next_state_values.cuda(self.device)
            a_m = a_m.cuda(self.device)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1,1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + self.gamma*next_state_values

        return expected_state_action_values


    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values,
                self.expected_state_action_values.unsqueeze(1))  

        if self.device != torch.device('cpu'):
            loss = loss.cuda(self.device)

        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()


    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    
    def update_td_error_memory(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        if self.device != torch.device('cpu'):
            state_batch = state_batch.cuda(self.device)
            action_batch = action_batch.cuda(self.device)
            reward_batch = reward_batch.cuda(self.device)
            non_final_next_states = non_final_next_states.cuda(self.device)


        state_action_values = self.main_q_network(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        
        next_state_values = torch.zeros(len(self.memory))
        a_m = torch.zeros(len(self.memory)).type(torch.LongTensor)

        if self.device != torch.device('cpu'):
            next_state_values = next_state_values.cuda(self.device)
            a_m = a_m.cuda(self.device)


        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]
        
        a_m_non_final_next_states = a_m[non_final_mask].view(-1,1)
        next_state_values[non_final_mask] = self.target_q_network(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        td_errors = (reward_batch + self.gamma*next_state_values) - state_action_values.squeeze()

        self.td_error_memory.memory = td_errors.detach().cpu().numpy().tolist()

        
    def reset_noise(self):
        self.main_q_network.reset_noise()
        self.target_q_network.reset_noise()


class Agent:

    def __init__(self, num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, td_epsilon, is_training, device):

        self.brain = Brain(num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, td_epsilon, is_training, device)

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()

    def reset_noise(self):
        self.brain.reset_noise()

