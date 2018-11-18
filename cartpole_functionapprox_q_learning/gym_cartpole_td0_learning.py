import gym, sys, os
sys.path.append('..')
from gym import wrappers
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch, torch.optim as optim
import torch.nn as nn
import random
import collections
from collections import deque
import utils, copy
from tqdm import tqdm

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99 # discount factor
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cpu')

EPOCHS = 500
DATA_COLLECTION_NUM_EPISODES = 100
TRAIN_NUM_BATCHES_PER_EPOCH = 200
TEST_NUM_EPISODES = 50

LAMBDA = 0.01    # speed of decay
MAX_EPSILON = 1
MIN_EPSILON = 0.01 # stay a bit curious even when getting old

# experience replay buffer
class Memory:
    def __init__(self, size):
        self.size = size
        self.samples = deque()
    
    def append(self, x):
        # reduce the size until it become self.size 
        if isinstance(x, collections.Iterable):
            # if it is array, the add it
            in_items_len = len(x)
            while (len(x) + in_items_len) >= self.size:
                x.popleft()
            
            self.samples += x
        else:
            # if it is single element, append it
            while (len(x) + 1) >= self.size:
                x.popleft()
            
            self.samples.append(x)
    
    def sample(self, n):
        # sample random n samples
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

class Brain(nn.Module):
    """Core logic of DQN
    """

    def __init__(self, nStateDim, nActions):
        super().__init__()

        # an MLP for state-action value function
        self.state_action_value = nn.Sequential(
            nn.Linear(nStateDim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nActions)
        )

    def forward(self, x):
        # if np array, convert into torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(DEVICE).float()

        # if state has no batch size, create a dummy batch with one sample
        if x.ndimension() == 1:
            x = x.unsqueeze(0)

        return self.state_action_value(x)
    
class Agent:
    def __init__(self, env, nBufferSize=MEMORY_CAPACITY):
        self.env = env
        self.brain = Brain(len(env.observation_space.sample()), env.action_space.n).to(DEVICE).float()
        self.memory = Memory(nBufferSize)

        # optimizer setup
        self.optimizer = optim.SGD(self.brain.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        self.steps = 0

    def act(self, s, epsilon):
        '''epsilon greedy action selection
        
        Arguments:
            s {np array} -- state selection
            epsilon {float} -- epsilon value
        
        Returns:
            action -- action index
        '''

        if epsilon > np.random.uniform():
            action = self.env.action_space.sample()
        else:  
            with torch.no_grad():
                q = self.brain(s)
                action = torch.argmax(q, dim=1).item()

        return action
    
    def get_epsilon(self):
        '''get epsilon value
        
        Returns:
            float -- epsilon value
        '''

        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        self.steps += 1
        return self.epsilon

    def play_one_episode(self, is_test=False):
        '''play one episode of cart pole
        
        Keyword Arguments:
            is_test {bool} -- is it a test episode? (default: {False})
        
        Returns:
            samples, total_reward, iters -- collected samples, rewards and total iterations
        '''

        current_state = env.reset()
        total_reward = 0
        iters = 0
        samples = []
        done = False

        while not done:
            # sample an action according to epsilon greedy strategy
            epsilon = 0
            if not is_test:
                epsilon = self.get_epsilon()
            
            action = self.act(current_state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            if done:
                if iters < 198:
                    reward = -300   
        
                next_state = None

            iters += 1
            total_reward += reward
            samples.append([current_state, action, reward, next_state])
            current_state = next_state

        return samples, total_reward, iters

    def collect_data(self, datacol_num_episodes):
        '''collect data and put it into replay memory
        
        Arguments:
            datacol_num_episodes {int} -- number of episodes to run and collect data
        
        Returns:
            avg_reward -- average reward in data collection
        '''

        total_datacol_rewards = 0
        total_train_loss = 0

        for i in range(datacol_num_episodes):
            # play each episode and put the samples into memory
            samples, total_reward, iters = self.play_one_episode()
            self.memory.append(samples)
            total_datacol_rewards += total_reward
        
        avg_reward = total_datacol_rewards / datacol_num_episodes
        return avg_reward

    def get_target_Q_values(self, rewards, next_states, model):
        '''get the target Q values for Q-learning
        
        Arguments:
            rewards {tensor} -- reward for each transition
            next_states {array of list} -- state description
            model {DQ network} -- target model
        
        Returns:
            Q target -- Q target values
        '''

        targetQvals = torch.zeros(len(next_states), 1).to(DEVICE)

        for i, next_state in enumerate(next_states):
            if next_state is None:
                targetQvals[i, 0] = rewards[i]
            else:
                state_torch = torch.Tensor(next_state).float().to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_torch)
                
                max_q_value = torch.max(q_values, dim=1)[0].item()
                targetQvals[i, 0] = rewards[i] + GAMMA * max_q_value

        return targetQvals
    
    def tensorize_samples(self, samples):
        '''collect the current states, rewards, actions, next states from the sampled data from 
        experience replay buffer
        
        Arguments:
            samples {array of list} -- data sampled from replay buffer
        
        Returns:
            current_states, actions, rewards, next_states -- separated data
        '''

        current_states, actions, rewards, next_states = [],[],[],[]
        for _, data in enumerate(samples):
            current_states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])

        current_states = np.array(current_states, dtype=np.float)
        current_states = torch.from_numpy(current_states).to(DEVICE).float()
        actions = torch.from_numpy(np.array(actions)).to(DEVICE).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float)).to(DEVICE).float().unsqueeze(1)
        return current_states, actions, rewards, next_states

    def train(self, num_runs = 100):
        '''train the DQN for num_runs iterations
        
        Keyword Arguments:
            num_runs {int} -- number of iterations to train (default: {100})
        
        Returns:
            avg_train_loss -- average train loss
        '''

        # backup the brain for fixed target
        target_model = copy.deepcopy(self.brain)
        total_loss = 0
        self.brain.train()

        for i in range(num_runs):
            # sample data from memory
            data = self.memory.sample(BATCH_SIZE)
            current_states, actions, rewards, next_states = self.tensorize_samples(data)

            # get max Q values of next state to form fixed target
            target_Q = self.get_target_Q_values(rewards, next_states, target_model)

            # update the model
            currentQ = self.brain(current_states)
            self.optimizer.zero_grad()
            loss = torch.mean((target_Q - currentQ.gather(1, actions))**2)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
    
        return total_loss / num_runs

    def test(self, num_episodes):
        '''test the model
        
        Arguments:
            num_episodes {int} -- number of episodes to run
        
        Returns:
            avg_reward -- average reward
        '''

        total_rewards = 0
        self.brain.eval()

        for _ in range(num_episodes):
            samples, total_reward, iters = self.play_one_episode(is_test=True)
            total_rewards += total_reward
        
        avg_rewards = total_rewards / num_episodes
        return avg_rewards

# calculate moving average
def moving_average (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = Agent(env)
    vis = utils.get_visdom_for_current_run('funcapprox_td0_qlearning')

    # average meters for losses and rewards
    avg_datacoll_reward_meter = utils.AverageMeter(vis, 'data collection rewards', 'epoch', 'rewards')
    avg_train_loss_meter = utils.AverageMeter(vis, 'train loss', 'epoch', 'loss')
    avg_test_reward_meter = utils.AverageMeter(vis, 'test rewards', 'epoch', 'rewards')
    
    for epoch in tqdm(range(EPOCHS)):
        # in each epoch, collect data
        average_data_collection_reward = model.collect_data(DATA_COLLECTION_NUM_EPISODES)
        # train the network
        average_train_loss = model.train(TRAIN_NUM_BATCHES_PER_EPOCH)
        # test the network
        average_test_reward = model.test(TEST_NUM_EPISODES)

        # logistics
        print(average_data_collection_reward, average_train_loss, average_test_reward)
        avg_datacoll_reward_meter.update(average_data_collection_reward, DATA_COLLECTION_NUM_EPISODES)
        avg_train_loss_meter.update(average_train_loss, TRAIN_NUM_BATCHES_PER_EPOCH)
        avg_test_reward_meter.update(average_test_reward, TEST_NUM_EPISODES)    

    # save video, if needed
    if 'monitor' in sys.argv:
        print('plotting and saving the video')
        filename = os.path.basename(__file__).split(".")[0]
        monitor_dir = "../scratch/" + filename + "_" + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    
    average_test_reward = model.test(1)
    print('reward in testing: ', average_test_reward)
    env.env.close()
    env.close()
