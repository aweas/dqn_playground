import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class QNetworkTf():
    """Actor (Policy) Model."""

    def __init__(self, session, state_size, action_size, name):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.sess = session
        self.name = name

        self.input = tf.placeholder(tf.float32, shape=(None, state_size))
        self.y_input = tf.placeholder(tf.float32, shape=(None, 1))

        self.gather_index = tf.placeholder(tf.int32, shape=(None))

        self.output = self._inference(state_size, action_size)
        self.loss, self.optimizer = self._training_graph()

        self.sess.run([tf.global_variables_initializer(),
                       tf.local_variables_initializer()])

    def _inference(self, state_size, action_size):
        with tf.variable_scope("inference_"+self.name):
            layer = tf.layers.dense(self.input, 64, activation=tf.nn.relu)
            layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
            layer = tf.layers.dense(layer, 4)
        return layer

    def _training_graph(self):
        with tf.variable_scope('training_'+self.name):
            pad = tf.range(tf.size(self.gather_index))
            pad = tf.expand_dims(pad, 1)
            ind = tf.concat([pad, self.gather_index], axis=1)

            gathered = tf.gather_nd(self.output, ind)
            gathered = tf.expand_dims(gathered, 1)
            loss = tf.losses.mean_squared_error(labels=self.y_input, predictions=gathered)
            # loss = tf.reduce_mean(loss)

            optimize = tf.train.AdamOptimizer(
                learning_rate=5e-4).minimize(loss)

        return loss, optimize

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.sess.run(self.output, feed_dict={self.input: state})

    def train(self, states, y_correct, actions):
        ls, _ =  self.sess.run([self.loss, self.optimizer], feed_dict={
                      self.input: states, self.y_input: y_correct, self.gather_index: actions})
        return ls
