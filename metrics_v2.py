"""Metrics for measuring machine learning algorithm performances
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        #y_pred_ = tf.to_int32(y_pred > t)
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def iou(actual, predicted):
    """Compute Intersection over Union statistic (i.e. Jaccard Index)

    See https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Intersection over Union value
    """
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    intersection = K.sum(actual * predicted)
    union = K.sum(actual) + K.sum(predicted) - intersection
    print(intersection, type(intersection))
    print(union, type(union))
    return 1. * intersection / union

def iou_loss(actual, predicted):
    """Loss function based on the Intersection over Union (IoU) statistic

    IoU is comprised between 0 and 1, as a consequence the function is set as
    `f(.)=1-IoU(.)`: the loss has to be minimized, and is comprised between 0
    and 1 too

    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Intersection-over-Union-based loss
    """
    return 1. - iou(actual, predicted)

def dice_coef(actual, predicted, eps=1e-3):
    """Dice coef

    See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Examples at:
      -
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
      -
    https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py#L36


    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    eps : float
        Epsilon value to add numerical stability

    Returns
    -------
    float
        Dice coef value
    """
    y_true_f = K.flatten(actual)
    y_pred_f = K.flatten(predicted)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

def dice_coef_loss(actual, predicted):
    """
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Dice-coef-based loss
    """
    return -dice_coef(actual, predicted)

