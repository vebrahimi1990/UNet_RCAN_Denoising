import numpy as np
import tensorflow as tf
from keras.activations import sigmoid
from keras.layers import Dropout, LeakyReLU, UpSampling3D
from keras.layers.convolutional import Conv3D
from keras.layers.merge import concatenate, add, multiply
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D
from keras.models import Model


def kinit(size, filters):
    n = 1 / np.sqrt(size * size * filters)
    w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    return w_init


def CAB(inputs, filters_cab, filters, kernel, dropout):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    z = GlobalAveragePooling3D(data_format='channels_last', keepdims=True)(x)
    z = Conv3D(filters=filters_cab, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(z)
    z = LeakyReLU()(z)
    z = Conv3D(filters=filters, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(z)
    z = sigmoid(z)
    z = multiply([z, x])
    z = add([z, inputs])
    return z


def RG(inputs, num_CAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_CAB):
        x = CAB(x, filters_cab, filters, kernel, dropout)
        x = Dropout(dropout)(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def RiR(inputs, num_RG, num_RCAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_RG):
        x = RG(x, num_RCAB, filters, filters_cab, kernel, dropout)
        x = Dropout(dropout)(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def make_RCAN(inputs, filters, filters_cab, num_RG, num_RCAB, kernel, dropout):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = RiR(x, num_RG, num_RCAB, filters, filters_cab, kernel, dropout)
    x = Conv3D(filters=1, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    return x


def make_generator(inputs, filters, num_filters, filters_cab, num_RG, num_RCAB, kernel_shape, dropout):
    y = make_RCAN(inputs=inputs, filters=num_filters, filters_cab=filters_cab, num_RG=num_RG, num_RCAB=num_RCAB,
                  kernel=kernel_shape, dropout=dropout)
    model = Model(inputs=[inputs], outputs=[y])
    return model
