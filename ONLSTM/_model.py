import numpy as np
import tensorflow as tf
from ._activation import cumax

__all__ = ['ONLSTM', 'split_point']

def split_point(x):
    x = x.numpy()
    return np.where(x == 1)[0].min()





class ONLSTM(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_unit):
        super(ONLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_unit = hidden_unit


    def build(self):
        # Equation 1-5: basic LSTM model
        self.w_f = self.add_weight(shape=(self.input_dim, self.hidden_unit), name='standard_lstm_kernel')
        self.w_i = self.add_weight(shape=(self.input_dim, self.hidden_unit), name='lstm_input_kernel')
        self.w_o = self.add_weight(shape=(self.input_dim, self.hidden_unit), name='lstm_output_kernel')
        self.w_c = self.add_weight(shape=(self.input_dim, self.hidden_unit), name='cell_memory_kernel')

        self.u_f = self.add_weight(shape=(self.hidden_unit, self.hidden_unit), name='standard_recurrent_kernewl')  # u_f
        self.u_i = self.add_weight(shape=(self.hidden_unit, self.hidden_unit), name='lstm_input_recurrent_kernel')
        self.u_o = self.add_weight(shape=(self.hidden_unit, self.hidden_unit), name='lstm_input_recurrent_kernel')
        self.u_c = self.add_weight(shape=(self.hidden_unit, self.hidden_unit), name='lstm_input_recurrent_kernel')

        self.b_f = self.add_weight(shape=(1, self.hidden_unit), name='forget_bias')
        self.b_i = self.add_weight(shape=(1, self.hidden_unit), name='input_bias')
        self.b_o = self.add_weight(shape=(1, self.hidden_unit), name='output_bias')
        self.b_c = self.add_weight(shape=(1, self.hidden_unit), name='cell_bias')

        # Trainable parameter of Equation 9
        self.w_fm = self.add_weight(shape=(self.input_dim, self.hidden_unit), name='onlstm_kernel')  # w_f
        self.u_fm = self.add_weight(shape=(self.hidden_unit, self.hidden_unit), name='recurrent_kernel')  # u_f
        self.b_fm = self.add_weight(shape=(1, self.hidden_unit))
        self.b_im = self.add_weight(shape=(1, self.hidden_unit))

        # attribute
        self.built = True


    def _base_lstm_op(self, x, h):
        f_out = tf.nn.sigmoid(tf.tensordot(x, self.w_f) + tf.tensordot(h, self.u_f) + self.b_f)
        i_out = tf.nn.sigmoid(tf.tensordot(x, self.w_i) + tf.tensordot(h, self.u_i) + self.b_i)
        o_out = tf.nn.sigmoid(tf.tensordot(x, self.w_o) + tf.tensordot(h, self.u_o) + self.b_o)
        c_out = tf.nn.tanh(tf.tensordot(x, self.w_c) + tf.tensordot(h, self.u_c) + self.b_c)
        return o_out * tf.nn.tanh(c_out), f_out, i_out, c_out


    def call(self, x, h):
        h, outputs = self._base_lstm_op(x, h)

        lstm_f = outputs[0]
        lstm_i = outputs[1]
        lstm_c_hat = outputs[2]


        # Equation 9: master forget gate.
        wx = tf.einsum('ij,jk->ik', x, self.w_f)  # (1xinput_dim) (input_dim x hidden) -> (1xhidden)
        uh = tf.einsum('ij,jk->ik', h, self.u_f)  # (1xhidden) (hidden x hidden) ->  (1 x hidden)
        f_t = cumax(wx + uh + self.b_f)  # (1 x hidden)

        # Equation 10: master input gate.
        i_t = 1 - cumax(wx + uh + self.b_i)  # (1 x hidden)

        # Equation 11:
        w_t = f_t * i_t #  (1 x hidden)

        # Equation 12:
        f_t_hat = f_t * (lstm_f * i_t + 1 - i_t)

        # Equation 13:
        i_t_hat = i_t * (lstm_i * f_t + 1 - f_t)

        # Equation 14: first d_t neuron of the previous cell state will be complete erased.
        c_t = f_t_hat * self.c_t + i_t_hat + lstm_c_hat
        self.c_t = c_t

        return h