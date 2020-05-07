import random
import torch
import numpy as np

from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = torch.cat([i[0] for i in batch], axis=0)
        actions= torch.cat([i[1] for i in batch], axis=0)
        rewards= torch.cat([i[2] for i in batch], axis=0)
        next_states = torch.cat([i[3] for i in batch], axis=0)
        dones  = [i[4] for i in batch]
        return states, actions, rewards, next_states, dones

    def buffer_size(self):
        return self.count
