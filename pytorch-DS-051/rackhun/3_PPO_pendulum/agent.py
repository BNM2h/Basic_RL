import os
import gym
import numpy as np
import torch

from model import Actor, Critic


class Agent:

    def __init__(self, env, gamma, gae_lambda, batch_size, lr_rate, ratio_clipping, epochs):

        self.env = env
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.epochs = epochs

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, lr_rate[0], ratio_clipping)
        self.critic = Critic(self.state_dim, lr_rate[1])

        self.save_epi_reward = []

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards)
        gae_cumulative = 0.
        forward_val = 0.

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma*forward_val - v_values[k]
            gae_cumulative = self.gamma*self.gae_lambda*gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]

        return gae, n_step_targets 
       
    def unpack_batch(self, batch):
        unpack = []
        for idx in range(len(batch)):
            unpack.append(batch[idx])

        unpack = torch.cat(unpack ,axis=0)
        return unpack
 
    def train(self, max_episode_num, save_path, save_names):
        batch_state, batch_action, batch_reward = [],[],[]
        batch_log_old_policy_pdf = []

        for episode in range(max_episode_num):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor)

            while not done:
                #env.render()
                mu_old, std_old, action = self.actor.get_policy_action(state)
                action  = np.array([action.item()])
                mu_old  = np.array([mu_old.item()])
                std_old = np.array([std_old.item()])
                action  = np.clip(action, -self.action_bound, self.action_bound)

                var_old = std_old**2
                log_old_policy_pdf = -0.5*(action-mu_old)**2/var_old - 0.5*np.log(var_old*2*np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                action     = torch.from_numpy(action).type(torch.FloatTensor)
                reward     = torch.FloatTensor([reward])
                log_old_policy_pdf = torch.FloatTensor([log_old_policy_pdf])

                state      = state.view(1, self.state_dim)
                next_state = next_state.view(1, self.state_dim)
                action     = action.view(1, self.action_dim)
                reward     = reward.view(1, 1)
                log_old_policy_pdf = log_old_policy_pdf.view(1,1)

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward+8)/8)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                if len(batch_state) < self.batch_size:
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                states  = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                batch_state, batch_action, batch_reward = [],[],[]
                batch_log_old_policy_pdf = []

                v_values     = self.critic.get_value(states)
                next_v_value = self.critic.get_value(next_state)
                gaes, y_i = self.gae_target(rewards, v_values, next_v_value, done)

                for _ in range(self.epochs):
                    self.actor.update(states, actions, gaes, log_old_policy_pdfs)
                    self.critic.update(states, y_i)
 

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

