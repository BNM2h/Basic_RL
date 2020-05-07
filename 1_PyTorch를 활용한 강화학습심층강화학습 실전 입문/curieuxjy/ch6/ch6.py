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

#----------------CONSTANTS-------------------
ENV = 'Cartpole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

#---------------memory class-------------------
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
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
#-----------------NN----------------------
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        # super? overriding?
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output
#----------------Brain Class----------------
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# CONST
BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        # Memory Object for transition
        self.memory = ReplayMemory(CAPACITY)

        # NN
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)# Net Class
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)

        # Optimizer
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):

        # transition 수 확인
        if len(self.memory) < BATCH_SIZE:
            return # nothing..?

        # minibatch
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # target Q(expected) 계산
        self.expected_state_action_values = self.get_expected_state_action_values()

        # weight update
        self.update_main_q_network()

    def decide_action(self, state, episode):
        # ε-greedy 
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # evaluation mode
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # max output의 index → max(1)[1]
            # [torch.LongTensor of size 1] 을 size 1*1로 변환 → .view(1,1)

        else:
            # random( 0 or 1 )
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
            # action은 [torch.LongTensor of size 1*1]

        return action

    def make_minibatch(self):
        # minibatch 추출
        transitions = self.memory.sample(BATCH_SIZE)

        # transitions → 각 step 별 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장
        #                = (state, action, state_next, reward) * BATCH_SIZE
        # to minibatch!
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환
        batch = Transition(*zip(*transitions))

        # state → [torch.FloatTensor of size 1*4]가 BATCH_SIZE 갯수만큼 
        #       → torch.FloatTensor of size BATCH_SIZE*4 형태로 변형
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        # switch to evaluation mode
        self.main_q_network.eval()
        self.target_q_network.eval()

        # NN으로 Q 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력 → [torch.FloatTensor of size BATCH_SIZEx2]
        # a_t에 대한 Q값을 계산
        # 1) action_batch에서 취한 행동 a_t가 왼쪽이냐 오른쪽이냐에 대한 index → ?
        # 2) 이에 대한 Q값을 gather로 모음
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # max{Q(s_t+1, a)}
        # !caution! 다음 상태가 존재하는지

        # NOT done, next_state가 존재하는지 확인하는 index mask
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        # intialize with 0
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 결합 가중치 수정'''

        # 4.1 신경망을 학습 모드로 전환
        self.main_q_network.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은
        # size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()  # 경사를 초기화
        loss.backward()  # 역전파 계산
        self.optimizer.step()  # 결합 가중치 수정

    def update_target_q_network(self):  # DDQN에서 추가됨
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
