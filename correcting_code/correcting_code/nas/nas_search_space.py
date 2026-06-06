import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# =========================================================
# OPERATIONS
# =========================================================

def op_conv3x3(filters):

    return tf.keras.Sequential([
        layers.Conv1D(
            filters,
            3,
            padding="same",
            activation="relu"
        )
    ])


def op_conv5x5(filters):

    return tf.keras.Sequential([
        layers.Conv1D(
            filters,
            5,
            padding="same",
            activation="relu"
        )
    ])


def op_skip():

    return tf.keras.Sequential([
        layers.Lambda(lambda x: x)
    ])


def op_maxpool():

    return tf.keras.Sequential([
        layers.MaxPooling1D(
            pool_size=3,
            strides=1,
            padding="same"
        )
    ])