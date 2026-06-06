from nas.nas_search_problem import DARTSNASProblem
from nas.nas_generic_problem import NASProblem
from nas.nas_search_space import MixedOp, DARTSSearchNetwork     

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class MixedOp(layers.Layer):

    def __init__(self, filters):

        super().__init__()

        self.ops = [
            op_conv3x3(filters),
            op_conv5x5(filters),
            op_skip(),
            op_maxpool(),
        ]

        self.alpha = tf.Variable(
            tf.random.normal([len(self.ops)]),
            trainable=True,
            name="alpha"
        )

    def call(self, x):

        weights = tf.nn.softmax(self.alpha)

        outputs = []

        for w, op in zip(weights, self.ops):

            outputs.append(
                w * op(x)
            )

        return tf.add_n(outputs)
    

class DARTSSearchNetwork(Model):

    def __init__(
        self,
        num_value_actions,
        num_position_actions,
        filters=128,
    ):

        super().__init__()

        self.stem = layers.Conv1D(
            filters,
            3,
            padding="same",
            activation="relu"
        )

        self.cell1 = MixedOp(filters)
        self.cell2 = MixedOp(filters)
        self.cell3 = MixedOp(filters)

        self.pool = layers.GlobalAveragePooling1D()

        # ============================================
        # VALUE HEAD
        # ============================================

        self.value_head = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_value_actions)
        ])

        # ============================================
        # POSITION HEAD
        # ============================================

        self.position_head = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_position_actions)
        ])

    def call(self, x):

        x = self.stem(x)

        x = self.cell1(x)
        x = self.cell2(x)
        x = self.cell3(x)

        x = self.pool(x)

        value_q = self.value_head(x)

        position_q = self.position_head(x)

        return value_q, position_q

    @property
    def arch_parameters(self):

        return [
            self.cell1.alpha,
            self.cell2.alpha,
            self.cell3.alpha,
        ]

    @property
    def weight_parameters(self):

        arch_ids = {
            id(p)
            for p in self.arch_parameters
        }

        return [
            p
            for p in self.trainable_variables
            if id(p) not in arch_ids
        ]
    

if __name__ == "__main__":

    train_dataset = ...
    valid_dataset = ...

    problem = DARTSNASProblem(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_value_actions=5,
        num_position_actions=3,
    )

    problem.search(
        epochs=20
    )

    architecture = problem.evaluate()

    print(architecture)