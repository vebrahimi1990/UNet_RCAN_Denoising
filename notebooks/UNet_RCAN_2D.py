from abc import ABC
import keras
from tensorflow.keras import layers, activations
import tensorflow as tf
tf.random.set_seed(42)



class CNN_block(layers.Layer):
    def __init__(self, filters, kernel, kernel_initializer, activation):
        super(CNN_block, self).__init__()
        self.activation = activation
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        self.ident = layers.Conv2D(filters=filters, kernel_size=1, padding='same', kernel_initializer=kernel_initializer)


    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        if self.activation == "leakyrelu":
            x = layers.LeakyReLU()(x)
        elif self.activation == "tanh":
            x = layers.Activation("tanh")(x)
        x = self.conv2(x)
        x = layers.add([x, self.ident(input_tensor)])
        x = layers.LeakyReLU()(x)
        return x


class CAB(layers.Layer):
    def __init__(self, filters_cab, filters, kernel, kernel_initializer, activation):
        super(CAB, self).__init__()
        self.activation = activation
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        self.cab_conv = layers.Conv2D(filters=filters_cab, kernel_size=1, padding='same', kernel_initializer=kernel_initializer)
        self.conv_cab = layers.Conv2D(filters=filters, kernel_size=1, padding='same', kernel_initializer=kernel_initializer)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        if self.activation == "leakyrelu":
            x = layers.LeakyReLU()(x)
        elif self.activation == "tanh":
            x = layers.Activation("tanh")(x)
        x = self.conv2(x)
        z = layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(x)
        z = self.cab_conv(z)
        z = layers.LeakyReLU()(z)
        z = self.conv_cab(z)
        z = activations.sigmoid(z)
        x = layers.multiply([x, z])
        x = layers.add([x, input_tensor])
        return x


class RG(layers.Layer):
    def __init__(self, num_cab, filters_cab, filters, kernel, kernel_initializer, activation):
        super(RG, self).__init__()
        self.cab = []
        for i in range(num_cab):
            self.cab.append(CAB(filters_cab, filters, kernel, kernel_initializer, activation))
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same", kernel_initializer=kernel_initializer)

    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.cab)):
            x = self.cab[i](x)
        x = self.conv(x)
        x = layers.add([x, input_tensor])
        return x


class RIR(layers.Layer):
    def __init__(self, num_RG, num_cab, filters_cab, filters, kernel, kernel_initializer, dropout, activation):
        super(RIR, self).__init__()
        self.drop = dropout
        self.RG = []
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        for i in range(num_RG):
            self.RG.append(RG(num_cab, filters_cab, filters, kernel, kernel_initializer, activation))

    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.RG)):
            x = self.RG[i](x)
            x = layers.Dropout(self.drop)(x)
        x = self.conv(x)
        x = layers.add([x, input_tensor])
        return x


class RCAN(layers.Layer):
    def __init__(self, num_RG, num_cab, filters_cab, filters, kernel, kernel_initializer, dropout, activation):
        super(RCAN, self).__init__()
        self.dropout = dropout
        self.rir = RIR(num_RG, num_cab, filters_cab, filters, kernel, kernel_initializer, dropout, activation)
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=kernel_initializer)
        self.conv2 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer=kernel_initializer)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.rir(x)
        x = self.conv2(x)
        return x


class UNet(layers.Layer):
    def __init__(self, filters, filters_cab, kernel, kernel_initializer, dropout, activation):
        super(UNet, self).__init__()
        self.encoder = []
        self.decoder = []
        self.cab = []
        self.drop = dropout
        self.activation = activation
        for i, f in enumerate(filters):
            self.encoder.append(CNN_block(f, kernel, kernel_initializer, activation))
        self.bridge1 = CNN_block(2 * filters[-1], kernel, kernel_initializer, activation)
        self.bridge2 = CNN_block(2 * filters[-1], kernel, kernel_initializer, activation)
        self.cnn_last = CNN_block(1, 1, kernel_initializer, activation)
        filters.reverse()
        for i, f in enumerate(filters):
            self.decoder.append(CNN_block(f, kernel, kernel_initializer, activation))
            self.cab.append(CAB(filters_cab, f, kernel, kernel_initializer, activation))

    def call(self, input_tensor):
        skip_encoder = []
        x = input_tensor
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x = layers.Dropout(self.drop)(x)
            skip_encoder.append(x)
            x = layers.MaxPooling2D(2)(x)
        x = self.bridge1(x)
        skip_encoder.append(x)
        x = self.bridge2(x)
        skip_encoder.reverse()
        for i in range(len(self.decoder)):
            x = layers.UpSampling2D(2, data_format='channels_last')(x)
            xs = self.cab[i](skip_encoder[i + 1])
            # xs = skip_encoder[i + 1]
            x = layers.concatenate([x, xs])
            x = self.decoder[i](x)
            x = layers.Dropout(self.drop)(x)
        x = self.cnn_last(x)
        return x


class UNet_RCAN(keras.Model, ABC):
    def __init__(self, model_config):
        super(UNet_RCAN, self).__init__()
        self.unet = UNet(sorted(model_config['filters']), model_config['filters_cab'], model_config['kernel'], model_config['kernel_initializer'], 
                         model_config['dropout'], model_config['activation'])
        self.rcan = RCAN(model_config['num_RG'], model_config['num_cab'], model_config['filters_cab'],
                         filters=model_config['filters'][0], kernel=model_config['kernel'], kernel_initializer=model_config['kernel_initializer'],
                         dropout=model_config['dropout'], activation=model_config['activation'])

    def call(self, input_tensor):
        x = input_tensor
        x = self.unet(x)
        y = layers.concatenate([x, input_tensor])
        y = self.rcan(y)
        return x, y
