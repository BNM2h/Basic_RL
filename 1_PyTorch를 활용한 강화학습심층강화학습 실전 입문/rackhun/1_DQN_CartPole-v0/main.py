import gym

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model import *

import warnings
warnings.filterwarnings(action='ignore')
gym.logger.set_level(40)


### Parameters Setting ###
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

BATCH_SIZE = 32
CAPACITY = 10000
LEARNING_RATE = 0.0001

SEED = 44
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

SAVE_MOVIE='./movie/cartpole_DQN.mp4'
#########################


def main():

    env = gym.make(ENV)
    env.seed(0)    # reproductivity
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = Agent(num_states, num_actions, CAPACITY, BATCH_SIZE, LEARNING_RATE, GAMMA, DEVICE)

    episode_10_list = np.zeros(10)
    episode_final = False
    frames = []

    for episode in range(NUM_EPISODES):
        #seed_everything(episode)
        observation = env.reset()
        state = torch.from_numpy(observation).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)

        for step in range(MAX_STEPS):

            #if episode_final is True:
                #frames.append(env.render(mode='rgb_array'))

            action = agent.get_action(state, episode)
            observation_next, _, done, _ = env.step(action.item())

            if done:
                state_next = None
                episode_10_list = np.hstack((episode_10_list[1:], step+1))

                if step < 195:
                    reward = torch.FloatTensor([-1.0])
                    complete_episodes = 0
                else:
                    reward = torch.FloatTensor([1.0])
                    complete_episodes += 1

            else:
                reward = torch.FloatTensor([0.0])
                state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

            agent.memorize(state, action, state_next, reward)
            agent.update_q_function()
            state = state_next

            if done:
                print('%d Episode: %d steps (average in recent 10: %.1lf)' %(episode, step+1, episode_10_list.mean()))
                break

        
        if episode_final is True:
            #display_frames_as_gif(frames, SAVE_MOVIE)
            break

        if complete_episodes >= 10:
            print('10 episodes consecutive success!')
            episode_final = True        




if __name__ == "__main__":
    main()

