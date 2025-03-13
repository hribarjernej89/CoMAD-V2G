import random
import math
import numpy as np
from collections import namedtuple
# The libs we need to construct the DQN agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# Import Replay Memory
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fp1 = nn.Linear(state_size, 128)
        self.fp2 = nn.Linear(128, 256)
        self.fp3 = nn.Linear(256, 256)
        self.fp4 = nn.Linear(256, 128)
        self.head_values = nn.Linear(128, 1)
        self.head_advantages = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fp1(state))
        x = F.relu(self.fp2(x))
        x = F.relu(self.fp3(x))
        x = F.relu(self.fp4(x))
        values = self.head_values(x)
        advantages = self.head_advantages(x)
        return values + (advantages - advantages.mean())

class DQNAgent(object):  # Hyper-parameters are set here
    def __init__(self, n_states, n_actions, batch_size=512, epsilon=.99,
                 epsilon_decay=.9998, epsilon_min=.25, tau=.001, memory_size=10000, learning_rate=.01,
                 gamma=.9, per_epsilon=.001, beta_start=.4, beta_inc=1.002, seed=404):

        self.num_actions = n_actions
        self.num_states = n_states
        self.random_seed = seed
        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        # Q-Network
        self.qnetwork_local = QNetwork(self.num_states, self.num_actions, self.random_seed).to(device)
        self.qnetwork_target = QNetwork(self.num_states, self.num_actions, self.random_seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(self.num_actions, self.memory_size, self.batch_size, self.random_seed)
        self.per_epsilon = per_epsilon

        self.q_episode_loss = []

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, x):
        # Ensure state is tensor form
        if type(x).__name__ == 'ndarray':
            state = torch.from_numpy(x).float().unsqueeze(0).to(device)
        else:
            state = x
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_actions))

    #
    # Storing experience in the replay memory
    #
    def remember(self, s, a, r, s_, d=False):
        self.memory.add(s, a, r, s_, d)

    #
    # Trains the Q model using experiences randomly sampled from the replay memory
    #
    def replay(self):
        if len(self.memory) > self.batch_size:
            #print("We are training the DQN Agent!")
            states, actions, rewards, next_states, probabilities, experiences_idx, dones = self.memory.sample()


            current_qs = self.qnetwork_local(states).gather(1, actions)
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.qnetwork_target(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs

            is_weights = np.power(probabilities * self.batch_size, -self.beta)
            is_weights = torch.from_numpy(is_weights / np.max(is_weights)).float().to(device)
            loss = (target_qs - current_qs).pow(2) * is_weights
            loss = loss.mean()
            # To track the loss over episode
            self.q_episode_loss.append(loss.detach().numpy())

            td_errors = (target_qs - current_qs).detach().numpy()
            self.memory.update_priorities(experiences_idx, td_errors, self.per_epsilon)

            self.qnetwork_local.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
            # ------------------ update exploration rate ------------------ #
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)

    #
    # Soft update of the target neural network
    #
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    #
    # Saves parameters of a trained Q model
    #
    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    #
    # Loads a saved Q model
    #
    def load(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path)), self.qnetwork_target.load_state_dict(torch.load(path))
        self.qnetwork_local.eval(), self.qnetwork_target.eval()

