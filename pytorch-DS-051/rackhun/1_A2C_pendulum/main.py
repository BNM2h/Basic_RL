import os
import gym

from utils import *
from model import *

import warnings
warnings.filterwarnings(action='ignore')
gym.logger.set_level(40)


### Parameters Setting ###
ENV = 'Pendulum-v0'

MAX_EPISODE = 1000

GAMMA = 0.95
BATCH_SIZE = 32
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
LR_RATE = [ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE]

SEED = 44

SAVE_FOLDER = './saved_model/'
SAVE_ACTOR  ='A2C_ACTOR.pth'
SAVE_CRITIC ='A2C_CRITIC.pth'
#########################

def main():

    env = gym.make(ENV)
    env.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = Agent(state_dim, action_dim, action_bound, BATCH_SIZE, LR_RATE, GAMMA)

    ### Training
    training(agent, env, state_dim, action_dim, action_bound)

    ### Visualize Results
    visualize(agent, env)



def visualize(agent, env):

    agent.actor.load(SAVE_FOLDER, SAVE_ACTOR)
    agent.critic.load(SAVE_FOLDER, SAVE_CRITIC)
 
    time = 0
    state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)

    while True:
        env.render()
        action = agent.actor.get_action(state)
        action = np.array([action.item()])

        state, reward, done, _ = env.step(action)
        time += 1
        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()


def training(agent, env, state_dim, action_dim, action_bound):

    save_epi_reward = []
    for episode in range(MAX_EPISODE):
        batch_state, batch_action, batch_td_target, batch_advantage = [],[],[],[]
        time, episode_reward, done = 0, 0, False
        state = env.reset()
        state = torch.from_numpy(state).type(torch.FloatTensor)

        while not done:
            #env.render()
            action = agent.get_action(state)
            action = np.array([action.item()])
            action = np.clip(action, -action_bound, action_bound)

            next_state, reward, done, _ = env.step(action)

            next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
            action     = torch.from_numpy(action).type(torch.FloatTensor)
            reward     = torch.FloatTensor([reward])

            state      = state.view(1, state_dim)
            next_state = next_state.view(1, state_dim)
            action     = action.view(1, action_dim)
            reward     = reward.view(1, 1)


            v_value = agent.critic_predict(state)
            next_v_value = agent.critic_predict(next_state)

            train_reward = (reward+8)/8
            advantage, y_i = agent.advantage_td_target(train_reward, v_value, next_v_value, done)

            batch_state.append(state)
            batch_action.append(action)
            batch_td_target.append(y_i)
            batch_advantage.append(advantage)

            if len(batch_state) < BATCH_SIZE:
                state = next_state[0]
                episode_reward += reward[0]
                time += 1
                continue

            states     = unpack_batch(batch_state)
            actions    = unpack_batch(batch_action)
            td_targets = unpack_batch(batch_td_target)
            advantages = unpack_batch(batch_advantage)

            batch_state, batch_action, batch_td_target, batch_advantage = [],[],[],[]

            agent.critic_train(states, td_targets)
            agent.actor_train(states, actions, advantages)

            state = next_state[0]
            episode_reward += reward[0]
            time += 1
        

        save_epi_reward.append(episode_reward.item())

        if len(save_epi_reward) < 20:      
            print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(save_epi_reward))
        else:
            print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(save_epi_reward[-20:]))


        if episode % 10 == 0:
            agent.actor.save(SAVE_FOLDER, SAVE_ACTOR)
            agent.critic.save(SAVE_FOLDER, SAVE_CRITIC)

                
if __name__ == "__main__":
    main()
