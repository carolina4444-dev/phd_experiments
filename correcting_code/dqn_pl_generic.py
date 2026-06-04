"""
Modular Deep Q-Learning Architecture
====================================

Implements the architecture shown in the diagram:

- CharConv encoder inspired by:
  "Natural Language Understanding from Scratch"
  (Zhang et al., Character-level ConvNet)

- Dual-network DQN:
    * Online QNetwork
    * Target QNetwork

- Multi-head Q outputs:
    * Value-selection Q head
    * Position-selection Q head

- Replay buffer
- Bellman target construction
- Transfer update to target network
- Modular RL problem interface

TensorFlow / Keras implementation.

===========================================================
INSTALL
===========================================================

pip install tensorflow numpy

===========================================================
DESIGN
===========================================================

Environment:
    Provides states and rewards.

Agent:
    Uses two actions:
        1. choose VALUE
        2. choose POSITION

State:
    Text/string sequence encoded with character convolutions.

Network:
    Character CNN encoder
    -> Dense decoder
    -> Two Q-heads

===========================================================
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DQNConfig:
    max_sequence_length: int = 256

    # Character vocabulary
    alphabet: str = (
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " .,;!?-_:/\\'\"@#$%^&*()+={}[]<>|\n\t"
    )

    embedding_dim: int = 16

    # Character ConvNet
    conv_filters: List[int] = None
    conv_kernel_sizes: List[int] = None
    conv_pool_sizes: List[int] = None

    dense_units: List[int] = None

    gamma: float = 0.99
    learning_rate: float = 1e-4

    replay_buffer_size: int = 100000
    batch_size: int = 32

    target_update_steps: int = 1000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000

    def __post_init__(self):
        # "Natural Language Understanding from Scratch"
        # Character ConvNet configuration
        if self.conv_filters is None:
            self.conv_filters = [256, 256, 256, 256, 256, 256]

        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [7, 7, 3, 3, 3, 3]

        if self.conv_pool_sizes is None:
            self.conv_pool_sizes = [3, 3, 0, 0, 0, 3]

        if self.dense_units is None:
            self.dense_units = [1024, 1024]


# =========================================================
# CHAR TOKENIZER
# =========================================================

class CharTokenizer:
    def __init__(self, alphabet: str, max_length: int):
        self.alphabet = alphabet
        self.max_length = max_length

        self.char_to_idx = {
            ch: i + 1 for i, ch in enumerate(alphabet)
        }

        self.vocab_size = len(self.char_to_idx) + 1

    def encode(self, text: str):
        text = text.lower()

        encoded = np.zeros(self.max_length, dtype=np.int32)

        for i, ch in enumerate(text[:self.max_length]):
            encoded[i] = self.char_to_idx.get(ch, 0)

        return encoded


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
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

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, value_actions, position_actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(value_actions),
            np.array(position_actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# ABSTRACT RL ENVIRONMENT
# =========================================================

class RLProblem:
    """
    Pluggable RL problem interface.
    """

    @property
    def num_value_actions(self) -> int:
        raise NotImplementedError

    @property
    def num_position_actions(self) -> int:
        raise NotImplementedError

    def reset(self) -> str:
        raise NotImplementedError

    def step(
        self,
        value_action: int,
        position_action: int,
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        raise NotImplementedError


# =========================================================
# EXAMPLE ENVIRONMENT
# =========================================================

class StringConstructionProblem(RLProblem):
    """
    Example environment.

    Goal:
        Construct target string by selecting:
            - character
            - insertion position
    """

    def __init__(self, target="hello world"):
        self.target = target
        self.charset = list("abcdefghijklmnopqrstuvwxyz ")

        self.max_len = len(target)

        self.reset()

    @property
    def num_value_actions(self):
        return len(self.charset)

    @property
    def num_position_actions(self):
        return self.max_len

    def reset(self):
        self.current = ""

        return self.current

    def step(self, value_action, position_action):

        char = self.charset[value_action]

        pos = min(position_action, len(self.current))

        self.current = (
            self.current[:pos]
            + char
            + self.current[pos:]
        )

        self.current = self.current[:self.max_len]

        reward = self._score()

        done = self.current == self.target

        return self.current, reward, done, {}

    def _score(self):

        score = 0

        for a, b in zip(self.current, self.target):
            if a == b:
                score += 1

        return score / len(self.target)


# =========================================================
# CHARACTER CNN ENCODER
# =========================================================

class CharCNNEncoder(tf.keras.Model):

    def __init__(self, config: DQNConfig, tokenizer: CharTokenizer):
        super().__init__()

        self.embedding = layers.Embedding(
            input_dim=tokenizer.vocab_size,
            output_dim=config.embedding_dim,
            input_length=config.max_sequence_length,
        )

        self.conv_blocks = []

        for filters, kernel_size, pool_size in zip(
            config.conv_filters,
            config.conv_kernel_sizes,
            config.conv_pool_sizes,
        ):

            block = []

            block.append(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                )
            )

            if pool_size > 0:
                block.append(
                    layers.MaxPooling1D(pool_size=pool_size)
                )

            self.conv_blocks.append(block)

        self.flatten = layers.Flatten()

    def call(self, x):

        x = self.embedding(x)

        for block in self.conv_blocks:
            for layer in block:
                x = layer(x)

        x = self.flatten(x)

        return x


# =========================================================
# DUAL-HEAD Q NETWORK
# =========================================================

class DualHeadQNetwork(tf.keras.Model):

    def __init__(
        self,
        config: DQNConfig,
        tokenizer: CharTokenizer,
        num_value_actions: int,
        num_position_actions: int,
    ):
        super().__init__()

        self.encoder = CharCNNEncoder(config, tokenizer)

        self.hidden_layers = []

        for units in config.dense_units:
            self.hidden_layers.append(
                layers.Dense(units, activation="relu")
            )
            self.hidden_layers.append(
                layers.Dropout(0.5)
            )

        # Q-head for VALUE actions
        self.value_head = layers.Dense(
            num_value_actions,
            activation=None,
            name="value_q_head",
        )

        # Q-head for POSITION actions
        self.position_head = layers.Dense(
            num_position_actions,
            activation=None,
            name="position_q_head",
        )

    def call(self, x):

        x = self.encoder(x)

        for layer in self.hidden_layers:
            x = layer(x)

        value_q = self.value_head(x)
        position_q = self.position_head(x)

        return value_q, position_q


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(
        self,
        env: RLProblem,
        config: DQNConfig,
    ):

        self.env = env
        self.config = config

        self.tokenizer = CharTokenizer(
            config.alphabet,
            config.max_sequence_length,
        )

        # ONLINE NETWORK
        self.q_network = DualHeadQNetwork(
            config,
            self.tokenizer,
            env.num_value_actions,
            env.num_position_actions,
        )

        # TARGET NETWORK
        self.target_network = DualHeadQNetwork(
            config,
            self.tokenizer,
            env.num_value_actions,
            env.num_position_actions,
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )

        self.replay_buffer = ReplayBuffer(
            config.replay_buffer_size
        )

        self.train_step_counter = 0

        self.update_target_network()

    # =====================================================
    # EPSILON POLICY
    # =====================================================

    def epsilon(self):

        progress = min(
            self.train_step_counter / self.config.epsilon_decay_steps,
            1.0,
        )

        eps = (
            self.config.epsilon_start
            + progress
            * (
                self.config.epsilon_end
                - self.config.epsilon_start
            )
        )

        return eps

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def act(self, state: str):

        encoded = self.tokenizer.encode(state)

        encoded = np.expand_dims(encoded, axis=0)

        if random.random() < self.epsilon():

            value_action = random.randint(
                0,
                self.env.num_value_actions - 1,
            )

            position_action = random.randint(
                0,
                self.env.num_position_actions - 1,
            )

            return value_action, position_action

        value_q, position_q = self.q_network(encoded)

        value_action = tf.argmax(value_q[0]).numpy()
        position_action = tf.argmax(position_q[0]).numpy()

        return value_action, position_action

    # =====================================================
    # TRAIN STEP
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

            # -----------------------------------------
            # Current Q-values
            # -----------------------------------------

            current_value_q, current_position_q = self.q_network(states)

            current_value_q = tf.reduce_sum(
                current_value_q
                * tf.one_hot(
                    value_actions,
                    self.env.num_value_actions,
                ),
                axis=1,
            )

            current_position_q = tf.reduce_sum(
                current_position_q
                * tf.one_hot(
                    position_actions,
                    self.env.num_position_actions,
                ),
                axis=1,
            )

            # -----------------------------------------
            # Target Q-values
            # -----------------------------------------

            next_value_q, next_position_q = self.target_network(
                next_states
            )

            max_next_value_q = tf.reduce_max(
                next_value_q,
                axis=1,
            )

            max_next_position_q = tf.reduce_max(
                next_position_q,
                axis=1,
            )

            target_value_q = (
                rewards
                + (1.0 - dones)
                * self.config.gamma
                * max_next_value_q
            )

            target_position_q = (
                rewards
                + (1.0 - dones)
                * self.config.gamma
                * max_next_position_q
            )

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            value_loss = tf.reduce_mean(
                tf.square(
                    target_value_q - current_value_q
                )
            )

            position_loss = tf.reduce_mean(
                tf.square(
                    target_position_q - current_position_q
                )
            )

            total_loss = value_loss + position_loss

        gradients = tape.gradient(
            total_loss,
            self.q_network.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.q_network.trainable_variables,
            )
        )

        self.train_step_counter += 1

        # ---------------------------------------------
        # TARGET NETWORK UPDATE
        # ---------------------------------------------

        if (
            self.train_step_counter
            % self.config.target_update_steps
            == 0
        ):
            self.update_target_network()

        return total_loss.numpy()

    # =====================================================
    # TARGET NETWORK TRANSFER
    # =====================================================

    def update_target_network(self):

        self.target_network.set_weights(
            self.q_network.get_weights()
        )

    # =====================================================
    # ENCODE
    # =====================================================

    def encode_state(self, state: str):

        return self.tokenizer.encode(state)

    # =====================================================
    # MAIN TRAINING LOOP
    # =====================================================

    def fit(
        self,
        episodes=1000,
        max_steps=100,
    ):

        for episode in range(episodes):

            state = self.env.reset()

            total_reward = 0

            for step in range(max_steps):

                value_action, position_action = self.act(state)

                next_state, reward, done, _ = self.env.step(
                    value_action,
                    position_action,
                )

                self.replay_buffer.push(
                    self.encode_state(state),
                    value_action,
                    position_action,
                    reward,
                    self.encode_state(next_state),
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

    config = DQNConfig()

    env = StringConstructionProblem(
        target="reinforcement"
    )

    agent = DQNAgent(
        env=env,
        config=config,
    )

    agent.fit(
        episodes=500,
        max_steps=50,
    )