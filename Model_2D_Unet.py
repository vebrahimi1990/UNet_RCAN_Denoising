import numpy as np
import tensorflow as tf
from keras.layers import Dropout, LeakyReLU, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate, add
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


def kinit(size, filters):
    n = 1 / np.sqrt(size * size * filters)
    w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    nit = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    return w_init


def kinit_bias(size, filters):
    # n = 1 / np.sqrt(size * size * filters)
    # w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    # w_init = 'random_normal'
    w_init = 'zeros'
    return w_init


def conv_block(inputs, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    y = Conv2D(filters=filters, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def make_generator(inputs, filters, num_filters, filters_cab, num_RG, num_RCAB, kernel_shape, dropout):
    skip_x = []
    skip_y = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = MaxPooling2D(2)(x)

    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_x.append(x)
    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_y.append(x)
    filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(filters):
        x = UpSampling2D(size=2, data_format='channels_last')(x)
        xs = skip_x[i + 1]
        x = concatenate([x, xs])
        x = conv_block(x, f, kernel_shape)
        skip_y.append(x)
        x = Dropout(dropout)(x)

    x = Conv2D(filters=1, kernel_size=1, kernel_initializer=kinit(3, filters[0]), bias_initializer=kinit(3, 1),
               padding="same")(x)
    skip_x.reverse()
    skip_y.reverse()
    model = Model(inputs=[inputs], outputs=[x])
    return model
