import numpy as np
import random
from collections import namedtuple, deque

from model import QNetworkTf

import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # self.TAU = tf.placeholder(tf.float32)

        self.qnetwork_local = QNetworkTf(self.sess, state_size, action_size, "local")
        self.qnetwork_target = QNetworkTf(self.sess, state_size, action_size, "target")

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

        self.soft_update_op = self._get_soft_update_op()

    def _get_soft_update_op(self):
        Qvars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_local')
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_target')


        assert len(Qvars) != 0
        assert len(target_vars) != 0
        return [tvar.assign(TAU*qvar + (1.0-TAU)*tvar) for qvar, tvar in zip(Qvars, target_vars)]

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self.qnetwork_local.forward([state])

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = np.expand_dims(np.amax(self.qnetwork_target.forward(next_states), 1), 1)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))
        self.qnetwork_local.train(states, Q_targets, actions)

        # ------------------- update target network ------------------- #
        self.soft_update(TAU)

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        self.sess.run(self.soft_update_op)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack(
            [e.next_state for e in experiences if e is not None])
        dones = np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
