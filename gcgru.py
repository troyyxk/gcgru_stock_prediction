import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from utils import calculate_laplacian
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh


class gcgru(AbstractRNNCell):

    def __init__(self, num_units, adj, num_gcn_nodes, s_index, **kwargs):
        super(gcgru, self).__init__(**kwargs)
        self.units = num_units
        self._gcn_nodes = num_gcn_nodes
        self.s_index = s_index
        adj = tf.sparse.to_dense(calculate_laplacian(adj), default_value=0)
        self._adj = adj

    @ property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # weights
        self.wz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wz')
        self.wr = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wr')
        self.wh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wh')

        self.w0 = self.add_weight(shape=(1, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='w0')

        # us
        self.uz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wz')
        self.ur = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='ur')
        self.uh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='uh')

        # biases
        self.bz = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bz")
        self.br = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="br")
        self.bh = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bh")
        self.built = True

    def call(self, inputs, states):

        # prev_output = states[0]
        # h = K.dot(inputs, self.kernel)
        # output = h + K.dot(prev_output, self.recurrent_kernel)
        # return output, output
        state = states[0]
        print("---state input----")
        print(state)
        print(tf.size(state))
        print(inputs)
        print(tf.size(inputs))
        print("---state input----")

#         tf.reshape(inputs,shape=[-1, self.units, self.units, self._gcn_nodes])
#         tf.reshape(inputs,shape=[-1, self.units, self.units, self._gcn_nodes])
        x = self.gc(inputs)

        print(self.wz)
        print(self.uz)
        print(x.shape)
        print(self.bz)

        z = K.dot(x, self.wz) + K.dot(x, self.uz) + self.bz
        z = sigmoid(z)
        r = K.dot(x, self.wr) + K.dot(x, self.ur) + self.br
        r = sigmoid(r)
        h = K.dot(x, self.wh) + K.dot((r*state), self.uh) + self.bh
        h = tanh(h)

        output = z * state + (1 - z) * h
        return output, output

    def gc(self, inputs):
        '''
            1 iteration of gcn, can have mutiple of them.
        '''
        ax = K.dot(inputs, self._adj)
        ax = ax[:, self.s_index]
        ax = tf.expand_dims(ax, -1)
        return K.dot(ax, self.w0)
