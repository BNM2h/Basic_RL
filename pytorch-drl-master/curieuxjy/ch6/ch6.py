import numpy as np
import matplotlib.pyplot as plt
import gym

from matplotlib import animation

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
  
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('DDQN-cartpole.gif', writer='imagemagick', fps=30, dpi=100)
    plt.show()
#---------namedtuple for Transition------------------
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
#---------CONSTANTS-------------------
ENV = 'Cartpole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500
#---------memory class-------------------
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1)% self.capacity
      
