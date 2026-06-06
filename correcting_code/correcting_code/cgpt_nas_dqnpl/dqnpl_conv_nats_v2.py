import os
import json
import random
from collections import deque

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers

from nats_bench import create


# =========================================================
# CONFIG
# =========================================================

NUM_EDGES = 6

# NATS-TSS operations
#
# 0 = none
# 1 = skip_connect
# 2 = nor_conv_1x1
# 3 = nor_conv_3x3
# 4 = avg_pool_3x3
#
NUM_OPS = 5

STATE_SIZE = NUM_EDGES + 1  # + edge_idx

GAMMA = 0.99
ALPHA = 1.0

LR = 1e-4

BATCH_SIZE = 32
REPLAY_SIZE = 50000

TARGET_UPDATE_FREQ = 10

EPISODES = 1000

SEED = 42

NATS_PATH = (
    "/data/ccarvalho/phd_working/"
    "cgpt_nas_experiemnts/benchmarks/"
    "NATS-tss-v1_0-3ffb9-simple"
)

RESULTS_DIR = "results_softq_policy"


# =========================================================
# REPRODUCIBILITY
# =========================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# =========================================================
# NATS UTILITIES
# =========================================================

def state_to_arch(state_edges):
    """
    Converts internal edge representation
    into NATS-TSS architecture tuple.

    Example:

    [3,1,2,0,4,3]
        ->
    (3,1,2,0,4,3)
    """

    return tuple(
        int(max(0, x))
        for x in state_edges
    )


# =========================================================
# ENVIRONMENT
# =========================================================

class NATSNASEnv:

    def __init__(self, api):

        self.api = api

        self.reset()

    def reset(self):

        self.edge_idx = 0

        self.edges = np.full(
            NUM_EDGES,
            -1,
            dtype=np.float32
        )

        return self._get_state()

    def _get_state(self):

        return np.concatenate(
            [
                self.edges,
                np.array(
                    [self.edge_idx],
                    dtype=np.float32
                )
            ]
        )

    def step(self, action):

        self.edges[self.edge_idx] = action

        self.edge_idx += 1

        done = self.edge_idx >= NUM_EDGES

        reward = 0.0

        if done:
            reward = self.evaluate()

        return (
            self._get_state(),
            reward,
            done,
            {}
        )

    def evaluate(self):

        arch = state_to_arch(
            self.edges
        )

        index = self.api.query_index_by_arch(
            arch
        )

        if index < 0:
            return 0.0

        info = self.api.get_more_info(
            index,
            dataset="cifar10",
            hp="200",
            is_random=False
        )

        acc = float(
            info["valid-accuracy"]
        )

        #
        # reward normalization
        #
        return acc / 100.0


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, size=REPLAY_SIZE):

        self.buffer = deque(
            maxlen=size
        )

    def add(self, transition):

        self.buffer.append(
            transition
        )

    def sample(self, batch_size):

        idx = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        return [
            self.buffer[i]
            for i in idx
        ]

    def __len__(self):

        return len(self.buffer)


# =========================================================
# MODEL
# =========================================================

def create_q_policy_network(
    state_size=STATE_SIZE,
    num_ops=NUM_OPS
):

    inp = layers.Input(
        shape=(state_size,)
    )

    dense0 = layers.Dense(
        512 * state_size
    )(inp)

    x = layers.Reshape(
        (
            512 * state_size,
            1
        )
    )(dense0)

    x = layers.Conv1D(
        256,
        7,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(
        3
    )(x)

    x = layers.Conv1D(
        256,
        7,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(
        3
    )(x)

    x = layers.Conv1D(
        256,
        3,
        activation="relu"
    )(x)

    x = layers.Conv1D(
        256,
        3,
        activation="relu"
    )(x)

    x = layers.Conv1D(
        256,
        3,
        activation="relu"
    )(x)

    x = layers.Conv1D(
        256,
        3,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(
        3
    )(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        1024,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.5
    )(x)

    x = layers.Dense(
        1024,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.5
    )(x)

    #
    # Q head
    #
    q_values = layers.Dense(
        num_ops,
        name="q_values"
    )(x)

    #
    # Policy head
    #
    policy_logits = layers.Dense(
        num_ops,
        name="policy_logits"
    )(x)

    return Model(
        inp,
        [
            q_values,
            policy_logits
        ]
    )


# =========================================================
# SOFT VALUE
# =========================================================

def soft_value(q_values):

    return (
        ALPHA
        *
        tf.reduce_logsumexp(
            q_values / ALPHA,
            axis=1
        )
    )


# =========================================================
# TRAIN STEP
# =========================================================

@tf.function
def train_step(
    model,
    target_model,
    optimizer,
    states,
    actions,
    rewards,
    next_states,
    dones
):

    with tf.GradientTape() as tape:

        q_values, logits = model(
            states,
            training=True
        )

        #
        # Q(s,a)
        #
        chosen_q = tf.reduce_sum(
            q_values
            *
            tf.one_hot(
                actions,
                NUM_OPS
            ),
            axis=1
        )

        #
        # Soft target
        #
        target_q_values, _ = target_model(
            next_states,
            training=False
        )

        next_v = soft_value(
            target_q_values
        )

        targets = (
            rewards
            +
            GAMMA
            *
            next_v
            *
            (1.0 - dones)
        )

        #
        # Q loss
        #
        q_loss = tf.reduce_mean(
            tf.square(
                targets
                -
                chosen_q
            )
        )

        #
        # Policy loss
        #
        probs = tf.nn.softmax(
            logits,
            axis=1
        )

        log_probs = tf.nn.log_softmax(
            logits,
            axis=1
        )

        policy_loss = tf.reduce_mean(
            tf.reduce_sum(
                probs
                *
                (
                    ALPHA
                    *
                    log_probs
                    -
                    q_values
                ),
                axis=1
            )
        )

        total_loss = (
            q_loss
            +
            policy_loss
        )

    grads = tape.gradient(
        total_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            grads,
            model.trainable_variables
        )
    )

    return (
        q_loss,
        policy_loss,
        total_loss
    )


# =========================================================
# TRAINING LOOP
# =========================================================

def train_softq_nas():

    api = create(
        NATS_PATH,
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSNASEnv(
        api
    )

    model = create_q_policy_network()

    target_model = create_q_policy_network()

    target_model.set_weights(
        model.get_weights()
    )

    optimizer = tf.keras.optimizers.Adam(
        LR
    )

    replay = ReplayBuffer()

    best_reward = -1.0
    best_arch = None

    reward_history = []
    architecture_history = []

    for ep in range(EPISODES):

        state = env.reset()

        episode_reward = 0.0

        while True:

            state_batch = state[
                None, :
            ].astype(
                np.float32
            )

            _, logits = model(
                state_batch,
                training=False
            )

            probs = tf.nn.softmax(
                logits[0]
            ).numpy()

            action = np.random.choice(
                NUM_OPS,
                p=probs
            )

            (
                next_state,
                reward,
                done,
                _
            ) = env.step(
                action
            )

            replay.add(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state

            episode_reward += reward

            #
            # Learn
            #
            if len(replay) >= BATCH_SIZE:

                batch = replay.sample(
                    BATCH_SIZE
                )

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                ) = zip(*batch)

                states = np.asarray(
                    states,
                    dtype=np.float32
                )

                actions = np.asarray(
                    actions,
                    dtype=np.int32
                )

                rewards = np.asarray(
                    rewards,
                    dtype=np.float32
                )

                next_states = np.asarray(
                    next_states,
                    dtype=np.float32
                )

                dones = np.asarray(
                    dones,
                    dtype=np.float32
                )

                train_step(
                    model,
                    target_model,
                    optimizer,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                )

            if done:
                break

        #
        # Target update
        #
        if ep % TARGET_UPDATE_FREQ == 0:

            target_model.set_weights(
                model.get_weights()
            )

        arch = state_to_arch(
            env.edges
        )

        reward_history.append(
            float(
                episode_reward
            )
        )

        architecture_history.append(
            list(arch)
        )

        if episode_reward > best_reward:

            best_reward = (
                episode_reward
            )

            best_arch = arch

        print(
            f"Episode {ep:4d} | "
            f"Reward={episode_reward:.4f} | "
            f"Best={best_reward:.4f} | "
            f"Arch={arch}"
        )

    return (
        model,
        best_arch,
        best_reward,
        reward_history,
        architecture_history
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    (
        model,
        best_arch,
        best_reward,
        reward_history,
        architecture_history
    ) = train_softq_nas()

    print()
    print(
        "BEST ARCH:",
        best_arch
    )

    print(
        "BEST REWARD:",
        best_reward
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    run_data = {

        "best_arch":
            list(best_arch),

        "best_reward":
            float(best_reward),

        "reward_history":
            [
                float(x)
                for x in reward_history
            ],

        "architecture_history":
            architecture_history
    }

    save_path = os.path.join(
        RESULTS_DIR,
        f"softq_policy_"
        f"{np.random.randint(1e9)}.json"
    )

    with open(
        save_path,
        "w"
    ) as f:

        json.dump(
            run_data,
            f,
            indent=2
        )

    print(
        f"Saved results to: "
        f"{save_path}"
    )
