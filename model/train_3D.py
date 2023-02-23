import tensorflow as tf
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Input
from tensorflow import keras
from config.config_3D import CFG
from data_preparation.data_generator import data_generator_3D
from model.UNet_RCAN_3D import UNet_RCAN
from model.loss import loss_3D

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data']
model_config = CFG['model']
callback = CFG['callbacks']
x_train, y_train = data_generator_3D(data_config)

model_input = Input((data_config['patch_size'], data_config['patch_size'],data_config['fr_end']-data_config['fr_start'], 1))
model = UNet_RCAN(model_config)

optimizer = keras.optimizers.Adam(learning_rate=model_config['lr'])
model.compile(optimizer=optimizer, loss=loss_3D)

callbacks = [
    EarlyStopping(patience=callback['patience_stop'], verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=callback['factor_lr'], patience=callback['patience_lr']),
    ModelCheckpoint(filepath=model_config['save_dr'], verbose=1, save_best_only=True, save_weights_only=True)]

with open(os.path.join(model_config['save_config'], '', 'configuration.txt'), 'w') as data:
    data.write(str(CFG['model']))

results = model.fit(x=x_train, y=y_train, batch_size=model_config['batch_size'],
                    epochs=model_config['n_epochs'],
                    verbose=1, callbacks=callbacks, validation_split=0.1)
