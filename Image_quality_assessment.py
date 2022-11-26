import tensorflow as tf


def norm_mse(prediction, gt):
    mse = tf.keras.metrics.mean_squared_error(prediction, gt)
    mse = tf.math.reduce_sum(mse, axis=(1, 2))
    norm = tf.norm(gt, axis=(1, 2))
    norm = tf.squeeze(norm)
    norm = tf.pow(norm, 2)
    norm = tf.math.reduce_sum(norm)
    nmse = tf.math.divide(mse, norm)
    return nmse.numpy()


def nmse_psnr_ssim(prediction, gt):
    nmse = norm_mse(prediction, gt)
    psnr = tf.image.psnr(prediction, gt, max_val=1.0).numpy()
    ssim = tf.image.ssim_multiscale(prediction, gt, max_val=1.0, filter_size=14,
                                    filter_sigma=1.5, k1=0.01, k2=0.03).numpy()
    return nmse, psnr, ssim
