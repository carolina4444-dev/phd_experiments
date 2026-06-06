
"""
softq_nas.py

Self-contained example of Soft Q-Learning NAS with editable-tree (position,value)
actions. Replace TokenAndPositionEmbedding/inception details as needed.

Requires:
    tensorflow
    numpy
    gym
    datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from collections import deque
import random
import gym
from gym import spaces
from datasets import load_dataset

# -------------------------------------------------
# Problem
# -------------------------------------------------

class NASProblem:
    def __init__(self, train_dataset, valid_dataset,
                 input_shape, max_length,
                 vocab_size, embedding_dim):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.input_shape = input_shape
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim


def build_agnews_problem():
    BATCH_SIZE = 64
    VALID_SIZE = 5000
    VOCAB_SIZE = 20000
    MAX_LENGTH = 250

    ds = load_dataset("ag_news")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH
    )

    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(train_texts).batch(256)
    )

    x_train = vectorizer(tf.constant(train_texts))
    y_train = tf.constant(train_labels)

    x_valid = x_train[:VALID_SIZE]
    y_valid = y_train[:VALID_SIZE]

    x_train = x_train[VALID_SIZE:]
    y_train = y_train[VALID_SIZE:]

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return NASProblem(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        input_shape=(MAX_LENGTH,),
        max_length=MAX_LENGTH,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128
    )

# -------------------------------------------------
# Child model
# -------------------------------------------------

def inception_module(x):
    b1 = layers.Conv1D(64, 1, padding="same", activation="relu")(x)

    b3 = layers.Conv1D(128, 1, padding="same", activation="relu")(x)
    b3 = layers.Conv1D(128, 3, padding="same", activation="relu")(b3)

    b5 = layers.Conv1D(32, 1, padding="same", activation="relu")(x)
    b5 = layers.Conv1D(32, 5, padding="same", activation="relu")(b5)

    bp = layers.MaxPooling1D(3, strides=1, padding="same")(x)
    bp = layers.Conv1D(32, 1, padding="same", activation="relu")(bp)

    return layers.Concatenate()([b1, b3, b5, bp])


def build_tree_model(problem, tree_encoding):
    inp = layers.Input(shape=problem.input_shape)

    x = layers.Embedding(
        problem.vocab_size,
        problem.embedding_dim
    )(inp)

    for bit in tree_encoding:
        if bit == 0:
            x = inception_module(x)
        elif bit == 1:
            x = layers.Conv1D(
                64, 3,
                padding="same",
                activation="relu"
            )(x)
            x = layers.MaxPooling1D(
                2,
                padding="same"
            )(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)

    out = layers.Dense(4, activation="softmax")(x)

    return Model(inp, out)

# -------------------------------------------------
# Environment
# -------------------------------------------------

class AGNewsNASEnvEditable(gym.Env):

    PAD = 4
    START = 3

    INCEPTION = 0
    CONV = 1
    STOP = 2

    def __init__(self, problem, max_nodes=20, train_epochs=1):
        super().__init__()

        self.problem = problem
        self.max_nodes = max_nodes
        self.train_epochs = train_epochs

        self.reset()

    def reset(self):
        self.state = np.full(
            self.max_nodes,
            self.PAD,
            dtype=np.int32
        )
        self.state[0] = self.START
        self.steps = 0
        return self.state.copy()

    def build_encoding(self):
        enc = []
        for v in self.state[1:]:
            if v in [0, 1]:
                enc.append(int(v))
        return enc

    def evaluate_architecture(self):

        tree = self.build_encoding()

        if len(tree) == 0:
            return 0.0

        model = build_tree_model(
            self.problem,
            tree
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        hist = model.fit(
            self.problem.train_dataset,
            validation_data=self.problem.valid_dataset,
            epochs=self.train_epochs,
            verbose=0
        )

        val_acc = max(hist.history["val_accuracy"])
        params = model.count_params()

        return float(
            val_acc - 1e-7 * params
        )

    def step(self, action):

        position, value = action

        done = False
        reward = 0.0

        if value == self.STOP:
            done = True
        else:
            if position > 0 and position < self.max_nodes:
                self.state[position] = value

        self.steps += 1

        if self.steps >= self.max_nodes:
            done = True

        if done:
            reward = self.evaluate_architecture()

        return self.state.copy(), reward, done, {}

# -------------------------------------------------
# Replay Buffer
# -------------------------------------------------

class ReplayBuffer:

    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------
# Soft Q Network
# -------------------------------------------------

def create_soft_q_network(
    state_size,
    max_nodes,
    num_values=3
):

    inp = layers.Input(shape=(state_size,))

    dense0 = layers.Dense(
        512 * state_size
    )(inp)

    x = layers.Reshape(
        (512 * state_size, 1)
    )(dense0)

    x = layers.Conv1D(256, 7, activation="relu")(x)
    x = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(256, 7, activation="relu")(x)
    x = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(256, 3, activation="relu")(x)
    x = layers.Conv1D(256, 3, activation="relu")(x)
    x = layers.Conv1D(256, 3, activation="relu")(x)
    x = layers.Conv1D(256, 3, activation="relu")(x)

    x = layers.MaxPooling1D(3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    pos_q = layers.Dense(max_nodes, name="position_q")(x)
    val_q = layers.Dense(num_values, name="value_q")(x)

    return Model(inp, [pos_q, val_q])

# -------------------------------------------------
# Agent
# -------------------------------------------------

def soft_value(q, alpha):
    return alpha * tf.reduce_logsumexp(
        q / alpha,
        axis=-1
    )


def train_softq_nas(
    episodes=100,
    max_nodes=20,
    alpha=1.0,
    gamma=0.99,
    batch_size=32
):

    problem = build_agnews_problem()

    env = AGNewsNASEnvEditable(
        problem,
        max_nodes=max_nodes,
        train_epochs=1
    )

    model = create_soft_q_network(
        max_nodes,
        max_nodes
    )

    target = create_soft_q_network(
        max_nodes,
        max_nodes
    )

    target.set_weights(model.get_weights())

    opt = tf.keras.optimizers.Adam(1e-4)

    replay = ReplayBuffer()

    for episode in range(episodes):

        state = env.reset()
        total_reward = 0.0

        while True:

            s = state.reshape(1, -1)

            pos_q, val_q = model(s, training=False)

            pos_probs = tf.nn.softmax(
                pos_q[0] / alpha
            ).numpy()

            val_probs = tf.nn.softmax(
                val_q[0] / alpha
            ).numpy()

            position = np.random.choice(
                max_nodes,
                p=pos_probs
            )

            value = np.random.choice(
                3,
                p=val_probs
            )

            next_state, reward, done, _ = env.step(
                (position, value)
            )

            replay.add(
                (
                    state,
                    position,
                    value,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state
            total_reward += reward

            if len(replay) >= batch_size:

                batch = replay.sample(batch_size)

                states, positions, values, rewards, next_states, dones = zip(*batch)

                states = np.asarray(states, dtype=np.float32)
                next_states = np.asarray(next_states, dtype=np.float32)

                rewards = np.asarray(rewards, dtype=np.float32)
                dones = np.asarray(dones, dtype=np.float32)

                positions = np.asarray(positions)
                values = np.asarray(values)

                with tf.GradientTape() as tape:

                    pos_q, val_q = model(
                        states,
                        training=True
                    )

                    pos_policy = tf.nn.softmax(
                        pos_q / alpha,
                        axis=-1
                    )

                    val_policy = tf.nn.softmax(
                        val_q / alpha,
                        axis=-1
                    )

                    pos_entropy = -tf.reduce_sum(
                        pos_policy *
                        tf.math.log(
                            pos_policy + 1e-8
                        ),
                        axis=1
                    )

                    val_entropy = -tf.reduce_sum(
                        val_policy *
                        tf.math.log(
                            val_policy + 1e-8
                        ),
                        axis=1
                    )

                    entropy_bonus = tf.reduce_mean(
                        pos_entropy + val_entropy
                    )

                    npos_q, nval_q = target(
                        next_states,
                        training=False
                    )

                    next_v = (
                        soft_value(npos_q, alpha)
                        +
                        soft_value(nval_q, alpha)
                    )

                    target_q = (
                        rewards +
                        gamma * next_v * (1.0 - dones)
                    )

                    chosen_pos = tf.reduce_sum(
                        pos_q *
                        tf.one_hot(
                            positions,
                            max_nodes
                        ),
                        axis=1
                    )

                    chosen_val = tf.reduce_sum(
                        val_q *
                        tf.one_hot(
                            values,
                            3
                        ),
                        axis=1
                    )

                    pred_q = chosen_pos + chosen_val

                    q_loss = tf.reduce_mean(
                        tf.square(
                            target_q - pred_q
                        )
                    )

                    entropy_coef = 0.01

                    loss = (
                        q_loss
                        - entropy_coef * entropy_bonus
                    )

                grads = tape.gradient(
                    loss,
                    model.trainable_variables
                )

                opt.apply_gradients(
                    zip(
                        grads,
                        model.trainable_variables
                    )
                )

            if done:
                break

        if episode % 10 == 0:
            target.set_weights(
                model.get_weights()
            )

        print(
            f"Episode={episode} "
            f"Reward={total_reward:.4f}"
        )

    return model


if __name__ == "__main__":
    train_softq_nas(
        episodes=100,
        max_nodes=20
    )