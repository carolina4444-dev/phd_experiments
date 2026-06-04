"""
Minimal DARTS-style implementation in TensorFlow/Keras
-----------------------------------------------------

This example demonstrates:
1. A searchable cell with mixed operations
2. Architecture parameters (alphas)
3. Differentiable relaxation of operation selection
4. Bi-level optimization loop

This is a pedagogical implementation, not a production NAS framework.

Tested with:
- TensorFlow 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# ============================================================
# Candidate operations
# ============================================================

def conv3x3(filters):
    return keras.Sequential([
        layers.Conv2D(filters, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ])


def conv5x5(filters):
    return keras.Sequential([
        layers.Conv2D(filters, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ])


def maxpool3x3():
    return keras.Sequential([
        layers.MaxPool2D(pool_size=3, strides=1, padding="same")
    ])


def identity():
    return keras.Sequential([
        layers.Lambda(lambda x: x)
    ])


# ============================================================
# Mixed Operation (core DARTS idea)
# ============================================================

class MixedOp(layers.Layer):
    """
    Weighted mixture of candidate operations.
    Architecture parameters alpha determine operation importance.
    """

    def __init__(self, filters):
        super().__init__()

        self.ops = [
            conv3x3(filters),
            conv5x5(filters),
            maxpool3x3(),
            identity()
        ]

        self.num_ops = len(self.ops)

        # Architecture parameters
        self.alpha = tf.Variable(
            tf.random.normal([self.num_ops], stddev=1e-3),
            trainable=True,
            name="alpha"
        )

    def call(self, x, training=False):

        # Softmax relaxation
        weights = tf.nn.softmax(self.alpha)

        outputs = []

        for w, op in zip(tf.unstack(weights), self.ops):
            outputs.append(w * op(x, training=training))

        return tf.add_n(outputs)


# ============================================================
# Search Cell
# ============================================================

class SearchCell(layers.Layer):

    def __init__(self, filters):
        super().__init__()

        self.op1 = MixedOp(filters)
        self.op2 = MixedOp(filters)

    def call(self, x, training=False):

        h1 = self.op1(x, training=training)
        h2 = self.op2(h1, training=training)

        return h2


# ============================================================
# DARTS Network
# ============================================================

class DARTSModel(keras.Model):

    def __init__(self, num_classes=10, filters=32):
        super().__init__()

        self.stem = keras.Sequential([
            layers.Conv2D(filters, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.cell1 = SearchCell(filters)
        self.cell2 = SearchCell(filters)

        self.pool = layers.GlobalAveragePooling2D()

        self.classifier = layers.Dense(num_classes)

    def call(self, x, training=False):

        x = self.stem(x, training=training)

        x = self.cell1(x, training=training)
        x = self.cell2(x, training=training)

        x = self.pool(x)

        return self.classifier(x)

    # --------------------------------------------------------
    # Collect architecture parameters separately
    # --------------------------------------------------------

    @property
    def arch_parameters(self):

        params = []

        for layer in self.layers:
            if isinstance(layer, SearchCell):

                params.append(layer.op1.alpha)
                params.append(layer.op2.alpha)

        return params

    @property
    def weight_parameters(self):

        arch_ids = {id(v) for v in self.arch_parameters}

        return [
            v for v in self.trainable_variables
            if id(v) not in arch_ids
        ]


# ============================================================
# Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.squeeze()
y_test = y_test.squeeze()

batch_size = 64

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(5000).batch(batch_size)

valid_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
valid_ds = valid_ds.batch(batch_size)


# ============================================================
# Model + Optimizers
# ============================================================

model = DARTSModel(num_classes=10)

criterion = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

weight_optimizer = keras.optimizers.Adam(1e-3)

arch_optimizer = keras.optimizers.Adam(3e-4)


# ============================================================
# Metrics
# ============================================================

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
valid_acc_metric = keras.metrics.SparseCategoricalAccuracy()


# ============================================================
# DARTS training step
# ============================================================

@tf.function
def train_step(train_x, train_y, valid_x, valid_y):

    # --------------------------------------------------------
    # 1. Update network weights
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        logits = model(train_x, training=True)

        loss = criterion(train_y, logits)

    grads = tape.gradient(
        loss,
        model.weight_parameters
    )

    weight_optimizer.apply_gradients(
        zip(grads, model.weight_parameters)
    )

    train_acc_metric.update_state(train_y, logits)

    # --------------------------------------------------------
    # 2. Update architecture parameters
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        logits_valid = model(valid_x, training=True)

        arch_loss = criterion(valid_y, logits_valid)

    arch_grads = tape.gradient(
        arch_loss,
        model.arch_parameters
    )

    arch_optimizer.apply_gradients(
        zip(arch_grads, model.arch_parameters)
    )

    valid_acc_metric.update_state(valid_y, logits_valid)

    return loss, arch_loss


# ============================================================
# Training loop
# ============================================================

epochs = 5

valid_iter = iter(valid_ds.repeat())

for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}/{epochs}")

    for step, (train_x, train_y) in enumerate(train_ds):

        valid_x, valid_y = next(valid_iter)

        loss, arch_loss = train_step(
            train_x,
            train_y,
            valid_x,
            valid_y
        )

        if step % 100 == 0:
            print(
                f"step={step:04d} "
                f"train_loss={loss:.4f} "
                f"arch_loss={arch_loss:.4f}"
            )

    train_acc = train_acc_metric.result()
    valid_acc = valid_acc_metric.result()

    print(f"train_acc = {train_acc:.4f}")
    print(f"valid_acc = {valid_acc:.4f}")

    train_acc_metric.reset_state()
    valid_acc_metric.reset_state()


# ============================================================
# Extract discrete architecture
# ============================================================

print("\nLearned architecture:")

for i, alpha in enumerate(model.arch_parameters):

    probs = tf.nn.softmax(alpha).numpy()

    op_names = [
        "conv3x3",
        "conv5x5",
        "maxpool3x3",
        "identity"
    ]

    best_op = op_names[np.argmax(probs)]

    print(f"MixedOp {i}: {best_op}")