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
import torch.nn.functional as F
import torch.distributions as distributions

BATCH_SIZE = 64
GAMMA = 0.99 # discount factor
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cpu')

EPOCHS = 50
TRAIN_NUM_EPISODES_PER_EPOCH = 200
TEST_NUM_EPISODES = 50

# EPSILON decay
LAMBDA = 0.01    # speed of decay
MAX_EPSILON = 1
MIN_EPSILON = 0.01 # stay a bit curious even when getting old

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
    def __init__(self, env):
        self.env = env
        self.brain = Brain(len(env.observation_space.sample()), env.action_space.n).to(DEVICE).float()

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
        # get action logits
        action_logits = self.brain(s)

        # create a categorical distribution from logits
        categorical_distribution = distributions.Categorical(logits=action_logits)

        # sample actions according to the distribution
        actions = categorical_distribution.sample()
        # print(actions.shape)

        # collect relevant log probabilities
        relevant_log_probs = categorical_distribution.log_prob(actions)
        # print(relevant_log_probs.shape)

        return actions[0].item(), relevant_log_probs

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
            
            action, log_prob = self.act(current_state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            if done:
                # if iters < 198:
                    # reward = -300   
        
                next_state = None

            iters += 1
            total_reward += reward
            samples.append([current_state, action, log_prob, reward, next_state])
            current_state = next_state
        
        # go through the samples and calculated expected reward
        R = 0
        for sample in samples[::-1]:
            R = sample[3] + GAMMA * R
            sample[3] = R

        # samples = [current_state, action_index, log probability, ]
        return samples, total_reward, iters

    def collect_loss(self, samples):
        '''collect the current states, rewards, actions, next states from the sampled data from 
        experience replay buffer
        
        Arguments:
            samples {array of list} -- data sampled from replay buffer
        
        Returns:
            current_states, actions, rewards, next_states -- separated data
        '''

        current_states, actions, rewards, log_probs, next_states = [],[],[],[], []
        for _, data in enumerate(samples):
            current_states.append(data[0])
            actions.append(data[1])
            log_probs.append(data[2])
            rewards.append(data[3])
            next_states.append(data[4])

        rewards = np.array(rewards)
        # print(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        loss = 0
        for sample_index in range(len(samples)):
            loss += (log_probs[sample_index] * rewards[sample_index])

        return loss / len(samples)

    def train(self, num_runs = 100):
        '''train the DQN for num_runs iterations
        
        Keyword Arguments:
            num_runs {int} -- number of iterations to train (default: {100})
        
        Returns:
            avg_train_loss -- average train loss
        '''

        # backup the brain for fixed target
        total_loss = 0
        self.brain.train()

        for i in range(num_runs):
            # play one episode and collect the actions and log probs
            samples, total_reward, iters = self.play_one_episode()
            loss = -self.collect_loss(samples)

            # update the model
            self.optimizer.zero_grad()
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

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = Agent(env)
    vis = utils.get_visdom_for_current_run('funcapprox_policy_gradient')

    # average meters for losses and rewards
    avg_train_loss_meter = utils.AverageMeter(vis, 'train loss', 'epoch', 'loss')
    avg_test_reward_meter = utils.AverageMeter(vis, 'test rewards', 'epoch', 'rewards')
    
    for epoch in tqdm(range(EPOCHS)):
        # train the network
        average_train_loss = model.train(TRAIN_NUM_EPISODES_PER_EPOCH)
        # test the network
        average_test_reward = model.test(TEST_NUM_EPISODES)

        # logistics
        print(average_train_loss, average_test_reward)
        avg_train_loss_meter.update(average_train_loss, TRAIN_NUM_EPISODES_PER_EPOCH)
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
