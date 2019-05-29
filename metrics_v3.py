# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:40:13 2019

@author: polyc
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

# Define IoU metric
def mean_iou(y_true, y_pred):
    #y_pred = tf.to_int32(y_pred > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2, weights = tf.constant([0.01, 0.99]))
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def mean_iou_v2(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)