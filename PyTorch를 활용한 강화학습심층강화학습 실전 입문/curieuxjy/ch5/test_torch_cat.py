# -*- coding: utf-8 -*-

from collections import namedtuple
import random
import torch

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

state = [1, 2]
action = [0.1, 0.5]
state_next = [0, 0]
reward = [5, -5]
state = torch.tensor(state)
action = torch.tensor(action)
state_next = torch.tensor(state_next)
reward = torch.tensor(reward)
memory = []

for i in range(10):
  
  memory.append(Transition(state, action, state_next, reward))
  state += 1
  action += 0.1
  state_next += 0
  reward += 5


def sample(memory, batch_size):
  return random.sample(memory, batch_size)

sample_memory = sample(memory, 5)

batch = Transition(*zip(*sample_memory))
print(batch.state)

state_batch = torch.cat(batch.state)
action_batch = torch.cat(batch.action)
reward_batch = torch.cat(batch.reward)
non_final_next_states = torch.cat([s for s in batch.next_state
                                   if s is not None])