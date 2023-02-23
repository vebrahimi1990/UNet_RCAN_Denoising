import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Input
from tensorflow import keras
from UNet_RCAN_Denoising.config.config_2D import CFG
from UNet_RCAN_Denoising.data_preparation.data_generator import data_generator_2D
from UNet_RCAN_Denoising.model.UNet_RCAN_2D import UNet_RCAN
from UNet_RCAN_Denoising.model.loss import loss_2D

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data']
model_config = CFG['model']
callback = CFG['callbacks']
x_train, y_train = data_generator_2D(data_config)

model_input = Input((data_config['patch_size'], data_config['patch_size'], 1))
model = UNet_RCAN(model_config)

optimizer = keras.optimizers.Adam(learning_rate=model_config['lr'])
model.compile(optimizer=optimizer, loss=loss_2D)

callbacks = [
    EarlyStopping(patience=callback['patience_stop'], verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=callback['factor_lr'], patience=callback['patience_lr']),
    ModelCheckpoint(filepath=model_config['save_dr'], verbose=1, save_best_only=True, save_weights_only=True)]

results = model.fit(x=x_train[0:10], y=y_train[0:10], batch_size=model_config['batch_size'],
                    epochs=model_config['n_epochs'],
                    verbose=1, callbacks=callbacks, validation_split=0.1)
