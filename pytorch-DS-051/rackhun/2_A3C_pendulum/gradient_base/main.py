from agent import Agent

import warnings
warnings.filterwarnings(action='ignore')


### Parameters Setting ###
ENV = 'Pendulum-v0'

MAX_EPISODE = 1000
GAMMA = 0.95
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
LR_RATE = [ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE]
T_STEP_MAX = 4
ENTROPY_BETA = 0.01

#NUM_WORKERS = mp.cpu_count()
NUM_WORKERS = 4
SEED = 44

SAVE_FOLDER = './saved_model/'
SAVE_ACTOR  ='A3C_ACTOR.pth'
SAVE_CRITIC ='A3C_CRITIC.pth'
SAVE_NAMES = [SAVE_ACTOR, SAVE_CRITIC]

#########################

def main():

    ### A3C Global Agent
    agent = Agent(ENV, NUM_WORKERS)

    ### Training
    agent.train(MAX_EPISODE, LR_RATE, GAMMA, T_STEP_MAX, ENTROPY_BETA, SAVE_FOLDER, SAVE_NAMES)

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

    env.close()

                
if __name__ == "__main__":
    main()
