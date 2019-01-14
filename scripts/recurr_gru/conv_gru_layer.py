''' 
References:

Keras GRU
https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L422
Keras ConvLSTM2D
https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py

RFCNN 
https://gitlab.com/sepehr.valipour/RFCNN/blob/master/rfcnn/layers/convolutional.py
Literature at https://ieeexplore.ieee.org/abstract/document/8296851

TODO:
1. Get versions of packages; put in header
tensorflow-1.12.0
keras-applications-1.0.6                               
keras-preprocessing: 1.0.5

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

import tensorflow as tf


from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.keras.engine.base_layer import Layer
# from keras.layers.convolutional_recurrent import ConvRNN2D, ConvLSTM2D
from keras.legacy import interfaces
from keras.legacy.layers import Recurrent, ConvRecurrent2D
from keras.models import Sequential

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
import pylab as plt
from keras.layers.convolutional_recurrent import ConvLSTM2D, ConvRNN2D

class ConvGRU2DCell(Layer):
    """Cell class for the ConvGRU2D layer.
      # Arguments
          filters: Integer, the dimensionality of the output space
              (i.e. the number of output filters in the convolution).
          kernel_size: An integer or tuple/list of n integers, specifying the
              dimensions of the convolution window.
          strides: An integer or tuple/list of n integers,
              specifying the strides of the convolution.
              Specifying any stride value != 1 is incompatible with specifying
              any `dilation_rate` value != 1.
          padding: One of `"valid"` or `"same"` (case-insensitive).
          data_format: A string,
              one of `channels_last` (default) or `channels_first`.
              It defaults to the `image_data_format` value found in your
              Keras config file at `~/.keras/keras.json`.
              If you never set it, then it will be "channels_last".
          dilation_rate: An integer or tuple/list of n integers, specifying
              the dilation rate to use for dilated convolution.
              Currently, specifying any `dilation_rate` value != 1 is
              incompatible with specifying any `strides` value != 1.
          activation: Activation function to use.
              If you don't specify anything, no activation is applied
              (ie. "linear" activation: `a(x) = x`).
          recurrent_activation: Activation function to use
              for the recurrent step.
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix,
              used for the linear transformation of the inputs.
          recurrent_initializer: Initializer for the `recurrent_kernel`
              weights matrix,
              used for the linear transformation of the recurrent state.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
              the `kernel` weights matrix.
          recurrent_regularizer: Regularizer function applied to
              the `recurrent_kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          kernel_constraint: Constraint function applied to
              the `kernel` weights matrix.
          recurrent_constraint: Constraint function applied to
              the `recurrent_kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.
          dropout: Float between 0 and 1.
              Fraction of the units to drop for
              the linear transformation of the inputs.
          recurrent_dropout: Float between 0 and 1.
              Fraction of the units to drop for
              the linear transformation of the recurrent state.
          reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).
      """
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvGRU2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        print("kernel_size")
        print(self.kernel_size)
        
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
            

    def build(self, input_shape):
        print("building")

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters * 3,), 
                name='bias',
                initializer= self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :, :, :self.filters]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :, :, :self.filters]
        # reset gate
        self.kernel_r = self.kernel[:, :, :, self.filters: self.filters * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, :, :, self.filters:
                                                        self.filters * 2]
        # new gate                                                                                           self.filters * 3]
        self.kernel_h = self.kernel[:, :, :, self.filters * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, :, :, self.filters * 2:]

        if self.use_bias:
            # bias for inputs
            self.bias_z = self.bias[:self.filters]
            self.bias_r = self.bias[self.filters: self.filters * 2]
            self.bias_h = self.bias[self.filters * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

        
        
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
            self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        x_z = self.input_conv(inputs_z, self.kernel_z, self.bias_z,
                            padding=self.padding)
        x_r = self.input_conv(inputs_r, self.kernel_r, self.bias_r,
                            padding=self.padding)
        x_h = self.input_conv(inputs_h, self.kernel_h, self.bias_h,
                            padding=self.padding)
      
        h_z = self.recurrent_conv(h_tm1_z,
                                self.recurrent_kernel_z)
        h_r = self.recurrent_conv(h_tm1_r,
                                self.recurrent_kernel_r)

        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)

        h_h = self.recurrent_conv(r * h_tm1_h,
                                self.recurrent_kernel_h)

        hh = self.recurrent_activation(x_h + h_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                          padding=padding,
                          data_format=self.data_format,
                          dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                              data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                          padding='same',
                          data_format=self.data_format)
        return conv_out
    

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ConvGRU2D(ConvRNN2D):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        print("kernel_size")
        print(kernel_size)
        cell = ConvGRU2DCell(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          recurrent_activation=recurrent_activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          recurrent_initializer=recurrent_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          recurrent_regularizer=recurrent_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          recurrent_constraint=recurrent_constraint,
                          bias_constraint=bias_constraint,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout)
        print("created cell")
        super(ConvGRU2D, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        print("called")
        return super(ConvGRU2D, self).call(inputs, mask=mask,
                                           training=training, initial_state=initial_state)
    
    
    @property
    def filters(self):
        return self.cell.filters
    
    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'activity_regularizer': regularizers.serialize(
                  self.activity_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

""" 
From https://github.com/keras-team/keras/blob/master/examples/conv_lstm.py
To test the layer
"""


# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.


seq = Sequential()
seq.add(ConvGRU2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvGRU2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvGRU2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvGRU2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')
'''

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')
'''

# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# Train the network
noisy_movies, shifted_movies = generate_movies(n_samples=1200)
seq.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10,
        epochs=300, validation_split=0.05)

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    # plt.savefig('%i_animate.png' % (i + 1))



