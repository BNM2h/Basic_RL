import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
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

    anim.save('movie_cartpole_DQN')
    display(display_animation(anim, default_mode='loop'))

# Key-Value로 저장하는 nametuple
# from collection import nametuple

# Tr = nametuple('tr', ('name_a', 'value_b'))
# Tr_object = Tr('이름A', 100)

# print(Tr_object)
# print(Tr_object.value_b)

from collections import namedtuple

Transition = namedtuple(
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
        self.memory[self.index] = Transition(state, action, state_next, reward)

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
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))
        # to see the model structure
        print(self.model)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    def replay(self):
        # if size of transition is smaller than batch size, do NOTHING
        if len(self.memory)< BATCH_SIZE:
            return
        # get minibatch(self.memory = ReplayMemory object)
        transitions = self.memory.sample(BATCH_SIZE)
        # transformation (s,a,s_n,r)*batch -> (s*batch,a*batch,s_n*batch,r*batch)
        batch = Transition(*zip(*transitions))

        # reshaping into minibatch, make Variable for NN
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Evaluation Mode
        self.model.eval()


        # calculate Q(s_t, a_t)
        state_action_values = self.model(state_batch).gather(1, action_batch)
        # calculate max{Q(s_t+1, a)}
        # check that there is next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # Training Mode (update the weights)
        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # update the weights
        self.optimizer.zero_grad() #initialize
        loss.backward() #backpropagation
        self.optimizer.step() # update
    def decide_action(self, state, episode):
        # e-greedy algorithm
        epsilon = 0.5 * (1/(episode+1))

        if epsilon <= np.random.uniform(0,1):
            self.model.eval() # Evaluation Mode
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
        else:
            #random action
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]
            ) # action = [torch.LongTensor of size 1*1]
        return action

# Agent
class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    def update_q_function(self):
        self.brain.replay()
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

# Env
class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))
                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps: 최근 10 episode의 mean step = %.1f' % (
                        episode, step+1, episode_10_list.mean()
                    ))
                    break
            if episode_final is True:
                display_frames_as_gif(frames)
            if complete_episodes >= 10:
                print('10 episode success')
                episode_final = True

cartpole_env = Environment()
#cartpole_env.render()
cartpole_env.run()







        
