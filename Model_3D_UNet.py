import numpy as np
import tensorflow as tf
from keras.layers import Dropout, LeakyReLU, UpSampling3D
from keras.layers.convolutional import Conv3D
from keras.layers.merge import concatenate, add
from keras.layers.pooling import MaxPooling3D
from keras.models import Model


def kinit(size, filters):
    n = 1 / np.sqrt(size * size * filters)
    w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    return w_init


def conv_block(inputs, filters, kernel, dropout):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    y = Conv3D(filters=filters, kernel_size=3, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def make_generator(inputs, filters, num_filters, filters_cab, num_RG, num_RCAB, kernel_shape, dropout):
    skip_x = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_shape, dropout)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = MaxPooling3D((2, 2, 2))(x)

    x = conv_block(x, 2 * filters[-1], kernel_shape, dropout)
    filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(filters):
        x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        xs = skip_x[i]
        x = concatenate([x, xs])
        x = conv_block(x, f, kernel_shape, dropout)
        x = Dropout(dropout)(x)
    x = Conv3D(filters=1, kernel_size=1, kernel_initializer=kinit(3, filters[0]), bias_initializer=kinit(3, 1),
               padding="same")(x)

    model = Model(inputs=[inputs], outputs=[x])

    return model
