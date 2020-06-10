# COPIED FROM https://github.com/ikostrikov/pytorch-ddpg-naf
import random

import numpy as np
import math
import gym
import torch

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

#from models import Critic, Actor
#from common.replay_buffers import BasicBuffer
#from common.noise import OUNoise

class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.noise = OUNoise(self.env.action_space)
        
    def get_action(self, obs, eps=0.1):
        val = random.random()
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        if val <= eps:
            return np.random.uniform(-1.0, 1.0, size=action.shape)

        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
   
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        
        # update critic
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        # update actor
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_weight", torch.FloatTensor(num_out, num_in)) 
        self.register_buffer("epsilon_bias", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()

        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(self.epsilon_weight)) 
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(self.epsilon_bias))
        else:
            weight = self.mu_weight
            buas = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std,std)

        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()

    
class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out 
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out)) 
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_i", torch.FloatTensor(num_in))
        self.register_buffer("epsilon_j", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(epsilon_bias))
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_j = eps_j.sign() * (eps_j.abs()).sqrt()

# Ornstein-Ulhenbeck Noise
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[start]
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        eps = 0.1

        if (max_episodes - episode) // max_episodes <=  0.1:
            eps = 0.0
        
        for step in range(max_steps):
            action = agent.get_action(state, eps=eps)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            state = env.reset()
            episode_reward = 0
        
        state = next_state
            
    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean