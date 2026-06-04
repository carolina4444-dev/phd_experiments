"""
Updated Dual-Head Character-CNN DQN
===================================

Architecture EXACTLY adapted from the provided diagram.

Key differences from previous implementation:

INPUT PIPELINE
---------------
Input shape:
    (batch, 5)

Expanded to:
    (batch, 5, 5)

Flattened:
    (batch, 25)

Dense projection:
    25 -> 2560

Reshape:
    (batch, 5, 512)

Permute:
    (batch, 512, 5)

CHARACTER CNN
--------------
Conv1D(256, kernel=7)
MaxPool(3)

Conv1D(256, kernel=7)
MaxPool(3)

Conv1D(256, kernel=3)
Conv1D(256, kernel=3)
Conv1D(256, kernel=3)
Conv1D(256, kernel=3)

MaxPool(3)

Flatten

DUAL Q-HEADS
-------------
Head #1:
    Dense(1024)
    Dropout
    Dense(1024)
    Dropout
    Dense(value_actions)

Head #2:
    Dense(1024)
    Dropout
    Dense(1024)
    Dropout
    Dense(position_actions)

=========================================================
"""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tsp.dqn_pl_tsp_problem import TravelingSalesmanProblem


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:

    state_size: int = 5

    latent_dim: int = 512

    conv_filters: int = 256

    gamma: float = 0.99

    learning_rate: float = 1e-4

    batch_size: int = 32

    replay_size: int = 100000

    target_update_freq: int = 1000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        value_action,
        position_action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                state,
                value_action,
                position_action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        (
            states,
            value_actions,
            position_actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(value_actions),
            np.array(position_actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# ABSTRACT ENVIRONMENT
# =========================================================

class RLProblem:

    @property
    def num_value_actions(self):
        raise NotImplementedError

    @property
    def num_position_actions(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, value_action, position_action):
        raise NotImplementedError


# =========================================================
# EXAMPLE ENVIRONMENT
# =========================================================

class DummyProblem(RLProblem):

    """
    Example pluggable RL problem.
    """

    def __init__(self):

        self.state_dim = 5

        self.reset()

    @property
    def num_value_actions(self):
        return 5

    @property
    def num_position_actions(self):
        return 3

    def reset(self):

        self.state = np.random.rand(5)

        return self.state

    def step(self, value_action, position_action):

        reward = np.random.randn()

        next_state = np.random.rand(5)

        done = random.random() < 0.05

        self.state = next_state

        return next_state, reward, done, {}


# =========================================================
# MODEL
# =========================================================

def build_dual_head_q_network(
    state_size,
    num_value_actions,
    num_position_actions,
):

    inputs = layers.Input(shape=(state_size,))

    # =====================================================
    # Lambda
    # (?,5) -> (?,5,5)
    # =====================================================

    x = layers.Lambda(
        lambda t: tf.tile(
            tf.expand_dims(t, axis=-1),
            [1, 1, state_size]
        )
    )(inputs)

    # =====================================================
    # Lambda
    # (?,5,5) -> (?,25)
    # =====================================================

    x = layers.Lambda(
        lambda t: tf.reshape(
            t,
            (-1, state_size * state_size)
        )
    )(x)

    # =====================================================
    # Dense
    # (?,25) -> (?,2560)
    # =====================================================

    x = layers.Dense(
        2560,
        activation="relu"
    )(x)

    # =====================================================
    # Lambda
    # (?,2560) -> (?,5,512)
    # =====================================================

    x = layers.Lambda(
        lambda t: tf.reshape(
            t,
            (-1, 5, 512)
        )
    )(x)

    # =====================================================
    # Permute
    # (?,5,512) -> (?,512,5)
    # =====================================================

    x = layers.Permute((2, 1))(x)

    # =====================================================
    # CONV BLOCKS
    # =====================================================

    x = layers.Conv1D(
        256,
        kernel_size=7,
        activation="relu",
    )(x)

    x = layers.MaxPooling1D(
        pool_size=3
    )(x)

    x = layers.Conv1D(
        256,
        kernel_size=7,
        activation="relu",
    )(x)

    x = layers.MaxPooling1D(
        pool_size=3
    )(x)

    x = layers.Conv1D(
        256,
        kernel_size=3,
        activation="relu",
    )(x)

    x = layers.Conv1D(
        256,
        kernel_size=3,
        activation="relu",
    )(x)

    x = layers.Conv1D(
        256,
        kernel_size=3,
        activation="relu",
    )(x)

    x = layers.Conv1D(
        256,
        kernel_size=3,
        activation="relu",
    )(x)

    x = layers.MaxPooling1D(
        pool_size=3
    )(x)

    x = layers.Flatten()(x)

    # =====================================================
    # VALUE HEAD
    # =====================================================

    value_branch = layers.Dense(
        1024,
        activation="relu"
    )(x)

    value_branch = layers.Dropout(
        0.5
    )(value_branch)

    value_branch = layers.Dense(
        1024,
        activation="relu"
    )(value_branch)

    value_branch = layers.Dropout(
        0.5
    )(value_branch)

    value_output = layers.Dense(
        num_value_actions,
        name="value_q_values"
    )(value_branch)

    # =====================================================
    # POSITION HEAD
    # =====================================================

    position_branch = layers.Dense(
        1024,
        activation="relu"
    )(x)

    position_branch = layers.Dropout(
        0.5
    )(position_branch)

    position_branch = layers.Dense(
        1024,
        activation="relu"
    )(position_branch)

    position_branch = layers.Dropout(
        0.5
    )(position_branch)

    position_output = layers.Dense(
        num_position_actions,
        name="position_q_values"
    )(position_branch)

    return Model(
        inputs=inputs,
        outputs=[
            value_output,
            position_output
        ]
    )


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(
        self,
        env,
        config,
    ):

        self.env = env

        self.config = config

        self.q_network = build_dual_head_q_network(
            state_size=config.state_size,
            num_value_actions=env.num_value_actions,
            num_position_actions=env.num_position_actions,
        )

        self.target_network = build_dual_head_q_network(
            state_size=config.state_size,
            num_value_actions=env.num_value_actions,
            num_position_actions=env.num_position_actions,
        )

        self.target_network.set_weights(
            self.q_network.get_weights()
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )

        self.replay_buffer = ReplayBuffer(
            config.replay_size
        )

        self.train_steps = 0

    # =====================================================
    # EPSILON
    # =====================================================

    def epsilon(self):

        progress = min(
            self.train_steps / self.config.epsilon_decay,
            1.0
        )

        return (
            self.config.epsilon_start
            + progress
            * (
                self.config.epsilon_end
                - self.config.epsilon_start
            )
        )

    # =====================================================
    # ACTION
    # =====================================================

    def act(self, state):

        state = np.expand_dims(state, axis=0)

        if random.random() < self.epsilon():

            value_action = random.randint(
                0,
                self.env.num_value_actions - 1
            )

            position_action = random.randint(
                0,
                self.env.num_position_actions - 1
            )

            return value_action, position_action

        value_q, position_q = self.q_network(
            state,
            training=False
        )

        value_action = tf.argmax(
            value_q[0]
        ).numpy()

        position_action = tf.argmax(
            position_q[0]
        ).numpy()

        return value_action, position_action

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        if len(self.replay_buffer) < self.config.batch_size:
            return None

        (
            states,
            value_actions,
            position_actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.config.batch_size
        )

        with tf.GradientTape() as tape:

            value_q, position_q = self.q_network(
                states,
                training=True
            )

            chosen_value_q = tf.reduce_sum(
                value_q
                * tf.one_hot(
                    value_actions,
                    self.env.num_value_actions
                ),
                axis=1
            )

            chosen_position_q = tf.reduce_sum(
                position_q
                * tf.one_hot(
                    position_actions,
                    self.env.num_position_actions
                ),
                axis=1
            )

            next_value_q, next_position_q = self.target_network(
                next_states,
                training=False
            )

            max_next_value_q = tf.reduce_max(
                next_value_q,
                axis=1
            )

            max_next_position_q = tf.reduce_max(
                next_position_q,
                axis=1
            )

            target_value = (
                rewards
                + (1.0 - dones)
                * self.config.gamma
                * max_next_value_q
            )

            target_position = (
                rewards
                + (1.0 - dones)
                * self.config.gamma
                * max_next_position_q
            )

            value_loss = tf.reduce_mean(
                tf.square(
                    target_value
                    - chosen_value_q
                )
            )

            position_loss = tf.reduce_mean(
                tf.square(
                    target_position
                    - chosen_position_q
                )
            )

            loss = value_loss + position_loss

        gradients = tape.gradient(
            loss,
            self.q_network.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.q_network.trainable_variables
            )
        )

        self.train_steps += 1

        if (
            self.train_steps
            % self.config.target_update_freq
            == 0
        ):
            self.target_network.set_weights(
                self.q_network.get_weights()
            )

        return float(loss.numpy())

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        episodes=1000,
        max_steps=50,
    ):

        for episode in range(episodes):

            state = self.env.reset()

            total_reward = 0

            for step in range(max_steps):

                value_action, position_action = self.act(
                    state
                )

                (
                    next_state,
                    reward,
                    done,
                    _
                ) = self.env.step(
                    value_action,
                    position_action
                )

                self.replay_buffer.push(
                    state,
                    value_action,
                    position_action,
                    reward,
                    next_state,
                    done,
                )

                loss = self.train()

                state = next_state

                total_reward += reward

                if done:
                    break

            print(
                f"Episode={episode} "
                f"Reward={total_reward:.4f} "
                f"Loss={loss}"
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    config = Config()

    # env = DummyProblem()
    env = TravelingSalesmanProblem()

    agent = DQNAgent(
        env,
        config,
    )

    print(agent.q_network.summary())

    agent.fit(
        episodes=100,
        max_steps=25,
    )