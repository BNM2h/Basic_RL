import os
import gym
import torch
import numpy as np

from model import Actor, Critic
from replaybuffer import ReplayBuffer


class Agent:

    def __init__(self, env, gamma, batch_size, buffer_size, lr_rate, tau):

        self.env = env
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, lr_rate[0], tau)
        self.critic = Critic(self.state_dim, self.action_dim, lr_rate[1], tau)

        self.buffer = ReplayBuffer(self.buffer_size)
        self.save_epi_reward = []

    def ou_noise(self, x, rho=0.15, mu=0., dt=1e-1, sigma=0.2, dim=1):
        rho   = torch.FloatTensor([rho])
        mu    = torch.FloatTensor([mu])
        dt    = torch.FloatTensor([dt])
        return x + rho*(mu-x)*dt + torch.sqrt(dt)*torch.normal(0.,sigma, size=(dim,))
       
    def td_target(self, rewards, q_values, dones):
        y_k = torch.zeros(q_values.shape)

        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.gamma*q_values[i]
        return y_k
 
    def train(self, max_episode_num, save_path, save_names):
        self.actor.update_target_network()
        self.critic.update_target_network()

        for episode in range(max_episode_num):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor)

            pre_noise = torch.zeros(self.action_dim)

            while not done:
                #env.render()
                action  = self.actor.predict(state)[0]
                noise   = self.ou_noise(pre_noise, dim=self.action_dim)

                action  = np.array([action.item()])
                action  = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                action       = torch.from_numpy(action).type(torch.FloatTensor)
                reward       = torch.FloatTensor([reward])
                train_reward = torch.FloatTensor([(reward+8)/8])

                state        = state.view(1, self.state_dim)
                next_state   = next_state.view(1, self.state_dim)
                action       = action.view(1, self.action_dim)
                reward       = reward.view(1, 1)
                train_reward = reward.view(1, 1)
                 
                self.buffer.add_buffer(state, action, train_reward, next_state, done)
                if self.buffer.buffer_size > 1000:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

                    actions_  = self.actor.target_predict(next_states)
                    actions_  = actions_.view(next_states.shape[0], self.action_dim)
                    target_qs = self.critic.target_predict(next_states, actions_)
                    y_i = self.td_target(rewards, target_qs, dones)
                    self.critic.train(states, actions, y_i)

                    s_actions = self.actor.predict(states)
                    policy_loss = self.critic.predict(states, s_actions)
                    self.actor.train(policy_loss)

                    self.actor.update_target_network() 
                    self.critic.update_target_network()

                pre_noise = noise
                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            self.save_epi_reward.append(episode_reward.item())

            if len(self.save_epi_reward) < 20:
                print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(self.save_epi_reward))
            else:
                print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(self.save_epi_reward[-20:]))

            if episode % 10 == 0:
                self.actor.save(save_path, save_names[0])
                self.critic.save(save_path, save_names[1])

