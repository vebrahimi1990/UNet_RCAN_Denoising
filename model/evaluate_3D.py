import os
import numpy as np
import tensorflow as tf
from tifffile import imwrite
from UNet_RCAN_Denoising.config.config_3D import CFG
from UNet_RCAN_Denoising.data_preparation.data_generator import data_generator_3D
from UNet_RCAN_Denoising.model.UNet_RCAN_3D import UNet_RCAN

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data_test']
model_config = CFG['model']
x_test, y_test = data_generator_3D(data_config)

model = UNet_RCAN(model_config)
model(np.zeros(
    (1, data_config['patch_size'], data_config['patch_size'], data_config['fr_end'] - data_config['fr_start'], 1)))
model.load_weights(model_config['save_dr'])

prediction1 = np.zeros(x_test.shape)
prediction2 = np.zeros(x_test.shape)

for i in range(len(x_test)):
    prediction1[i], prediction2[i] = model(x_test[i:i + 1], training=False)
    prediction2[i] = prediction2[i] / prediction2[i].max()
prediction2[prediction2 < 0] = 0

pred2 = np.moveaxis(prediction2, 3, 1)
noisy = np.moveaxis(x_test, 3, 1)
gt = np.moveaxis(y_test, 3, 1)

pred2 = (pred2 * (2 ** 16 - 1)).astype(np.uint16)
noisy = (noisy * (2 ** 16 - 1)).astype(np.uint16)
gt = (gt * (2 ** 16 - 1)).astype(np.uint16)


imwrite(os.path.join(data_config['save_dr'], '', 'pred.tif'), pred2.squeeze(), imagej=True, metadata={'axes': 'TZYX'})
imwrite(os.path.join(data_config['save_dr'], '', 'noisy.tif'), noisy.squeeze(), imagej=True, metadata={'axes': 'TZYX'})
imwrite(os.path.join(data_config['save_dr'], '', 'gt.tif'), gt.squeeze(), imagej=True, metadata={'axes': 'TZYX'})
