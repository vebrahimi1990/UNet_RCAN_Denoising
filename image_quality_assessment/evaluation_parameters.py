import tensorflow as tf
import numpy as np


def norm_mse(prediction, gt):
    mse = tf.keras.metrics.mean_squared_error(prediction, gt)
    mse = tf.math.reduce_mean(mse, axis=(1, 2))
    norm = tf.norm(gt, axis=(1, 2))
    norm = tf.squeeze(norm)
    norm = tf.pow(norm, 2)
    nmse = tf.math.divide(mse, norm)
    return nmse.numpy()


def nmse_psnr_ssim_2D(prediction, gt):
    nmse = norm_mse(prediction, gt)
    psnr = tf.image.psnr(prediction, gt, max_val=1.0).numpy()
    ssim = tf.image.ssim_multiscale(prediction, gt, max_val=1.0, filter_size=14,
                                    filter_sigma=1.5, k1=0.01, k2=0.03).numpy()
    return nmse, psnr, ssim


def nmse_psnr_ssim_3D(prediction, gt):
    nmse = norm_mse(prediction, gt).mean(axis=-1)
    psnr = np.zeros((gt.shape[0], gt.shape[-2]))
    ssim = np.zeros((gt.shape[0], gt.shape[-2]))
    for i in range(gt.shape[-2]):
        psnr[:, i] = tf.image.psnr(prediction[:, :, :, i, :], gt[:, :, :, i, :], max_val=1.0).numpy()
        ssim[:, i] = tf.image.ssim_multiscale(prediction[:, :, :, i, :], gt[:, :, :, i, :], max_val=1.0, filter_size=14,
                                              filter_sigma=1.5, k1=0.01, k2=0.03)
    psnr = psnr.mean(axis=-1)
    ssim = ssim.mean(axis=-1)

    return nmse, psnr, ssim
