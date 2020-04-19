import os
import gym
import numpy as np

from model import Global_Actor,  Worker_Actor
from model import Global_Critic, Worker_Critic

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.multiprocessing as mp

global_episode_reward = []


class Agent:
  
    def __init__(self, env_name, n_workers, lr_rate, entropy_beta):

        self.env_name = env_name
        self.n_workers = n_workers

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        self.global_actor  = Global_Actor(state_dim, action_dim, action_bound, lr_rate[0], entropy_beta)
        self.global_critic = Global_Critic(state_dim, lr_rate[1])

        self.global_actor.share_memory()
        self.global_critic.share_memory()

        self.global_episode_count  = mp.Value('i', 0)
        self.global_step           = mp.Value('i', 0)

    def train(self, max_episode, gamma, t_max, path, names):
        workers = []
        for i in range(self.n_workers):
            worker_name = 'worker%i' % i
            workers.append(Worker(worker_name, self.env_name, self.global_actor, self.global_critic, max_episode, gamma, t_max, path, names, self.global_episode_count, self.global_step))

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()


class Worker(mp.Process):

    def __init__(self, name, env_name, global_actor, global_critic, max_episode, gamma, t_max, path, names, global_episode_count, global_step):
        super(Worker, self).__init__()

        self.worker_name = name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.max_episode = max_episode
        self.gamma = gamma
        self.t_max = t_max

        self.path = path
        self.actor_name  = names[0]
        self.critic_name = names[1]

        self.global_actor  = global_actor
        self.global_critic = global_critic

        self.worker_actor  = Worker_Actor(self.state_dim, self.action_dim, self.action_bound) 
        self.worker_critic = Worker_Critic(self.state_dim) 

        self.worker_actor.network.load_state_dict(self.global_actor.network.state_dict())
        self.worker_critic.network.load_state_dict(self.global_critic.network.state_dict())

        self.global_episode_count  = global_episode_count
        self.global_step           = global_step      

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = torch.zeros(rewards.size())
        cumulative = 0
 
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma*cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def unpack_batch(self, batch):
        unpack = []
        for idx in range(len(batch)):
            unpack.append(batch[idx])

        unpack = torch.cat(unpack ,axis=0)
        return unpack

    def run(self):
        global global_episode_reward
        print(self.worker_name, "starts ---")

        while self.global_episode_count.value <= int(self.max_episode):
            batch_state, batch_action, batch_reward = [],[],[]
            step, episode_reward, done = 0, 0, False
            state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor)

            while not done:
                #env.render()
                action = self.worker_actor.get_action(state)
                action = np.array([action.item()])
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                action     = torch.from_numpy(action).type(torch.FloatTensor)
                reward     = torch.FloatTensor([reward])

                state      = state.view(1, self.state_dim)
                next_state = next_state.view(1, self.state_dim)
                action     = action.view(1, self.action_dim)
                reward     = reward.view(1, 1)

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward+8)/8)

                state = next_state[0]
                episode_reward += reward[0]
                step += 1

                if len(batch_state) == self.t_max or done:
                    states  = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward) 
                    batch_state, batch_action, batch_reward = [],[],[]

                    v_values     = self.global_critic.get_value(states)
                    next_v_value = self.global_critic.get_value(next_state)
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    advantages = n_step_td_targets - v_values

                    self.global_critic.update(states, n_step_td_targets)
                    self.global_actor.update(states, actions, advantages)

                    self.worker_critic.network.load_state_dict(self.global_critic.network.state_dict())
                    self.worker_actor.network.load_state_dict(self.global_actor.network.state_dict())
                 
                    self.global_step.value += 1


                if done:
                    self.global_episode_count.value += 1
                    global_episode_reward.append(episode_reward.item())

                    if len(global_episode_reward) < 20:
                        print('Worker:', self.worker_name,\
                            ', Episode:', self.global_episode_count.value,\
                            ', Step:', step, ', Reward:', np.mean(global_episode_reward))
                    else:
                        print('Worker:', self.worker_name,\
                            ', Episode:', self.global_episode_count.value,\
                            ', Step:', step, ', Reward:', np.mean(global_episode_reward[-20:]))


                    if self.global_episode_count.value % 10 == 0:
                        self.global_actor.save(self.path, self.actor_name)
                        self.global_critic.save(self.path, self.critic_name)
                                        

