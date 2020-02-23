import random
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from utils import seed_everything

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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

    def __init__(self, num_states, num_actions, capacity, batch_size, lr_rate, gamma, device):
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_rate)


    def replay(self):

        #1. check memory
        if len(self.memory) < self.batch_size:
            return


        #2. mini batch 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).cuda()
        action_batch = torch.cat(batch.action).cuda()
        reward_batch = torch.cat(batch.reward).cuda()
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).cuda()

        next_state_values = torch.zeros(self.batch_size)

        #3. target Q
        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(self.batch_size).cuda()
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + self.gamma*next_state_values


        #4. Update
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = loss.cuda()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decide_action(self, state, episode):
        epsilon = 0.5*(1/(episode+1))
        state = state.cuda()

        if epsilon < np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action.cuda()

    
class Agent:

    def __init__(self, num_states, num_actions, capacity, batch_size, lr_rate, gamma, device):
        self.brain = Brain(num_states, num_actions, capacity, batch_size, lr_rate, gamma, device)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
