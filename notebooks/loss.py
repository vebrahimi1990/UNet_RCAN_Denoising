import numpy as np
import tensorflow as tf


def ch_loss_2D(pred, gt):
    norm = tf.norm(pred - gt, axis=(1, 2))
    norm = tf.squeeze(norm)
    norm = tf.pow(norm, 2)
    norm = norm / (256 * 256) + 1e-6
    norm = tf.pow(norm, 0.5)
    c_loss = tf.math.reduce_mean(norm)
    return c_loss


def edge_loss_2D(pred, gt):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape((3, 3, 1, 1))
    pred = tf.nn.conv2d(pred, kernel, strides=[1, 1, 1, 1], padding='VALID')
    gt = tf.nn.conv2d(gt, kernel, strides=[1, 1, 1, 1], padding='VALID')
    e_loss = ch_loss_2D(pred, gt)
    return e_loss


def loss_2D(prediction, gt):
    c_loss = ch_loss_2D(prediction, gt)
    e_loss = edge_loss_2D(prediction, gt)
    gen_loss = c_loss + 0.05 * e_loss
    return gen_loss


def ch_loss_3D(pred, gt):
    norm = tf.norm(tf.norm(pred - gt, axis=(1, 2)), axis=1)
    norm = tf.squeeze(norm)
    norm = tf.pow(norm, 2)
    norm = norm / (256 * 256) + 1e-6
    norm = tf.pow(norm, 0.5)
    c_loss = tf.math.reduce_mean(norm)
    return c_loss


def edge_loss_3D(pred, gt):
    pred = tf.transpose(pred, perm=[0, 3, 1, 2, 4])
    gt = tf.transpose(gt, perm=[0, 3, 1, 2, 4])
    kernel = np.array(
        [[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 2, 0], [2, -12, 2], [0, 2, 0]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]])
    kernel = kernel.reshape((3, 3, 3, 1, 1))
    pred = tf.nn.conv3d(pred, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    gt = tf.nn.conv3d(gt, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    e_loss = ch_loss_3D(pred, gt)
    return e_loss


def loss_3D(prediction, gt):
    c_loss = ch_loss_3D(prediction, gt)
    e_loss = edge_loss_3D(prediction, gt)
    gen_loss = c_loss + 0.05 * e_loss
    return gen_loss
