import random
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v   = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)   # No activation
        val = self.fc3_v(h2).expand(-1, adv.size(1))   # No activation

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output


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

    def __init__(self, num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, device):
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.main_q_network   = Net(num_states, hidden_size, self.num_actions)
        self.target_q_network = Net(num_states, hidden_size, self.num_actions)

        if self.device != torch.device('cpu'):
            self.main_q_network   = self.main_q_network.cuda(self.device)
            self.target_q_network = self.target_q_network.cuda(self.device)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr_rate)


    def replay(self):

        #1. check memory
        if len(self.memory) < self.batch_size:
            return

        #2. mini batch 
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        #3. target Q
        self.expected_state_action_values = self.get_expected_state_action_values()

        #4. Update
        self.update_main_q_network()


    def decide_action(self, state, episode):
        epsilon = 0.5*(1/(episode+1))

        if self.device != torch.device('cpu'):
            state = state.cuda(self.device)

        if epsilon < np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1,1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        if self.device == torch.device('cpu'):
            return action
        else:
            return action.cuda(self.device)


    def make_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
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



class Agent:

    def __init__(self, num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, device):
        self.brain = Brain(num_states, num_actions, capacity, batch_size, hidden_size, lr_rate, gamma, device)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

