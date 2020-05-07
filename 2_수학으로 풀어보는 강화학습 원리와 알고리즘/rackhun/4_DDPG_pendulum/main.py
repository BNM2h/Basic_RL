import gym
from agent import Agent

import warnings
warnings.filterwarnings(action='ignore')

### Parameters Setting ###
ENV = 'Pendulum-v0'

MAX_EPISODE = 200

GAMMA = 0.95
BATCH_SIZE = 64
BUFFER_SIZE = 20000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
LR_RATE = [ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE]
TAU = 0.001

SEED = 44

SAVE_FOLDER = './saved_model/'
SAVE_ACTOR  ='DDPG_ACTOR.pth'
SAVE_CRITIC ='DDPG_CRITIC.pth'
SAVE_NAMES = [SAVE_ACTOR, SAVE_CRITIC]
#############################


def main():
    env = gym.make(ENV)
    agent = Agent(env, GAMMA, BATCH_SIZE, BUFFER_SIZE, LR_RATE, TAU)

    ### Training
    agent.train(MAX_EPISODE, SAVE_FOLDER, SAVE_NAMES)


    ### Visualize Results
    #visualize(agent, env)

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


if __name__ == "__main__":
    main()
              
