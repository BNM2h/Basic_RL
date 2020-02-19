import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gym

# for animation
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('movie_cartpole_DQN.mp4')
    display(display_animation(anim, default_mode='loop'))

# Key-Value로 저장하는 nametuple
# from collection import nametuple

# Tr = nametuple('tr', ('name_a', 'value_b'))
# Tr_object = Tr('이름A', 100)

# print(Tr_object)
# print(Tr_object.value_b)

from collection import namedtuple

Transition = named(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

# Constants
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

# Class for Transition(Memory)
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # memory max capacity
        self.memory = [] # transition variable(list)
        self.index = 0 # save location(index)
    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) # NOT full memory

        # save memory using namedtuple Transition
        self.memory[self.index] = Transition(state, actino, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    # transition batch sampling 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # ? 그냥 길이를 재는 함수 아닌가?
    # get saved transition
    def __len__(self):
        return len(self.memory)

#-----------------------------------------------------------------------
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# constants
BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        # memory transition !OBJECT!
        self.memory = ReplayMemory(CAPACITY)

        # maek NN
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2')



        
