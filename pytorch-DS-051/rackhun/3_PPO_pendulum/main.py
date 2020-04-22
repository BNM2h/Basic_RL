import gym
from agent import Agent

import warnings
warnings.filterwarnings(action='ignore')

### Parameters Setting ###
ENV = 'Pendulum-v0'

MAX_EPISODE = 1000

GAMMA = 0.95
GAE_LAMBDA = 0.9
BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
LR_RATE = [ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE]
RATIO_CLIPPING = 0.2
EPOCHS = 10

SAVE_FOLDER = './saved_model/'
SAVE_ACTOR  ='PPO_ACTOR.pth'
SAVE_CRITIC ='PPO_CRITIC.pth'
SAVE_NAMES = [SAVE_ACTOR, SAVE_CRITIC]
#############################


def main():
    env = gym.make(ENV)
    agent = Agent(env, GAMMA, GAE_LAMBDA, BATCH_SIZE, LR_RATE, RATIO_CLIPPING, EPOCHS)

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
              
