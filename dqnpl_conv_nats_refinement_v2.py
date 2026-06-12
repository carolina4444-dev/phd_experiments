import os
import json
import random
from collections import deque

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers

from nats_bench import create
import matplotlib.pyplot as plt

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
OP_NAMES = {
    0: "none",
    1: "skip_connect",
    2: "nor_conv_1x1",
    3: "nor_conv_3x3",
    4: "avg_pool_3x3",
}

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

RESULTS_DIR = "results_softq_policy_refinement"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# REPRODUCIBILITY
# =========================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# Plotting utility

def save_reward_plot(reward_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("SoftQ NAS Reward Evolution")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward_curve.png"))
    plt.close()


def save_loss_plot(q_loss_hist, policy_loss_hist, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(q_loss_hist, label="Q Loss")
    plt.plot(policy_loss_hist, label="Policy Loss")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def save_architecture_heatmap(arch_history, output_dir):
    arr = np.array(arch_history, dtype=np.int32)

    plt.figure(figsize=(12, 6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar()
    plt.xlabel("Edge Index")
    plt.ylabel("Episode")
    plt.title("Architecture Evolution")
    plt.savefig(os.path.join(output_dir, "architecture_heatmap.png"))
    plt.close()


def save_best_arch(best_arch, best_reward, output_dir):
    plt.figure(figsize=(10, 2))
    plt.bar(np.arange(len(best_arch)), best_arch)
    plt.xlabel("Edge")
    plt.ylabel("Op")
    plt.title(f"Best Reward = {best_reward:.4f}")
    plt.savefig(os.path.join(output_dir, "best_architecture.png"))
    plt.close()

# =========================================================
# NATS UTILITIES
# =========================================================

def state_to_arch(state_edges):

    ops = [OP_NAMES[int(x)] for x in state_edges]

    arch = (
        f"|{ops[0]}~0|"
        f"+|{ops[1]}~0|{ops[2]}~1|"
        f"+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    )

    return arch


# =========================================================
# ENVIRONMENT
# =========================================================

class NATSNASEnv:

    def __init__(self, api):

        self.api = api
        self.reset()

    def reset(self):

        self.edges = np.full(
            NUM_EDGES,
            -1,
            dtype=np.int32
        )

        return self._get_state()

    def _get_state(self):

        assigned = np.sum(self.edges >= 0)

        return np.concatenate(
            [
                self.edges.astype(np.float32),
                np.array([assigned], dtype=np.float32)
            ]
        )

    def step(self, action):

        position = int(action[0])
        operation = int(action[1])

        #
        # invalid position
        #
        if position < 0 or position >= NUM_EDGES:

            return (
                self._get_state(),
                -1.0,
                True,
                {"error": "invalid_position"}
            )

        #
        # edge already assigned
        #
        if self.edges[position] != -1:

            return (
                self._get_state(),
                -0.1,
                False,
                {"error": "edge_already_filled"}
            )

        #
        # write operation
        #
        self.edges[position] = operation

        done = np.all(self.edges >= 0)

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

        #
        # safety check
        #
        if np.any(self.edges < 0):

            print(
                "Incomplete architecture:",
                self.edges
            )

            return 0.0

        arch = state_to_arch(self.edges)

        try:

            idx = self.api.query_index_by_arch(arch)

            info = self.api.get_more_info(
                idx,
                dataset="cifar10",
                hp="200",
                is_random=False
            )

            print("INFO KEYS:", info.keys())

            if "valid-accuracy" in info:
                acc = info["valid-accuracy"]

            elif "test-accuracy" in info:
                acc = info["test-accuracy"]

            elif "train-accuracy" in info:
                acc = info["train-accuracy"]

            else:
                raise RuntimeError(
                    f"Unknown keys: {list(info.keys())}"
                )

            reward = float(acc) / 100.0

            print(
                f"\nVALID ARCH\n"
                f"{arch}\n"
                f"reward={reward:.4f}\n"
            )

            return reward

        except Exception as e:

            print(
                "\nFAILED ARCH:",
                arch,
                "\n",
                e
            )

            return 0.0

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

def create_softq_network():

    inp = layers.Input(
        shape=(STATE_SIZE,)
    )

    dense0 = layers.Dense(
        512 * STATE_SIZE
    )(inp)

    x = layers.Reshape(
        (512 * STATE_SIZE, 1)
    )(dense0)

    x = layers.Conv1D(
        256,
        7,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(
        256,
        7,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(3)(x)

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

    x = layers.MaxPooling1D(3)(x)

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
    # Position Q-values
    #
    position_q = layers.Dense(
        NUM_EDGES,
        name="position_q"
    )(x)

    #
    # Operation Q-values
    #
    operation_q = layers.Dense(
        NUM_OPS,
        name="operation_q"
    )(x)

    return Model(
        inp,
        [
            position_q,
            operation_q
        ]
    )


# =========================================================
# SOFT VALUE
# =========================================================

def soft_value(position_q, operation_q):

    pos_v = (
        ALPHA
        * tf.reduce_logsumexp(
            position_q / ALPHA,
            axis=1
        )
    )

    op_v = (
        ALPHA
        * tf.reduce_logsumexp(
            operation_q / ALPHA,
            axis=1
        )
    )

    return pos_v, op_v


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

    pos_actions = actions[:, 0]
    op_actions = actions[:, 1]

    with tf.GradientTape() as tape:

        #
        # Current Q
        #
        pos_q, op_q = model(
            states,
            training=True
        )

        chosen_pos_q = tf.reduce_sum(
            pos_q
            * tf.one_hot(
                pos_actions,
                NUM_EDGES
            ),
            axis=1
        )

        chosen_op_q = tf.reduce_sum(
            op_q
            * tf.one_hot(
                op_actions,
                NUM_OPS
            ),
            axis=1
        )

        #
        # Target Q
        #
        next_pos_q, next_op_q = target_model(
            next_states,
            training=False
        )

        next_pos_v, next_op_v = soft_value(
            next_pos_q,
            next_op_q
        )

        target_pos = (
            rewards
            +
            GAMMA
            * next_pos_v
            * (1.0 - dones)
        )

        target_op = (
            rewards
            +
            GAMMA
            * next_op_v
            * (1.0 - dones)
        )

        #
        # Independent TD losses
        #
        pos_loss = tf.reduce_mean(
            tf.square(
                target_pos
                -
                chosen_pos_q
            )
        )

        op_loss = tf.reduce_mean(
            tf.square(
                target_op
                -
                chosen_op_q
            )
        )

        total_loss = (
            pos_loss
            +
            op_loss
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
        pos_loss,
        op_loss,
        total_loss
    )


# =========================================================
# TRAINING LOOP
# =========================================================

def train_softq_nas():

    reward_history = []
    architecture_history = []

    q_loss_history = []
    policy_loss_history = []
    total_loss_history = []

    api = create(
        NATS_PATH,
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSNASEnv(
        api
    )

    model = create_softq_network()

    target_model = create_softq_network()

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

            pos_q, op_q = target_model(
                state_batch,
                training=False
            )

            pos_probs = tf.nn.softmax(
                pos_q[0] / ALPHA
            ).numpy()

            op_probs = tf.nn.softmax(
                op_q[0] / ALPHA
            ).numpy()

            position = np.random.choice(
                NUM_EDGES,
                p=pos_probs
            )

            operation = np.random.choice(
                NUM_OPS,
                p=op_probs
            )

            action = (
                position,
                operation
            )

            (
                next_state,
                reward,
                done,
                _
            ) = env.step(
                action
            )

            replay.add((state, (position, operation), reward, next_state, done))

            state = next_state

            episode_reward += reward

            #
            # Learn
            #
            if len(replay) >= BATCH_SIZE:

                batch = replay.sample(
                    BATCH_SIZE
                )

                states, actions, rewards, next_states, dones = zip(*batch)

                actions = np.array(actions)

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

                q_loss, policy_loss, total_loss = train_step(
                    model,
                    target_model,
                    optimizer,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                )

                q_loss_history.append(float(q_loss.numpy()))
                policy_loss_history.append(float(policy_loss.numpy()))
                total_loss_history.append(float(total_loss.numpy()))

            if any(dones):
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

        reward_history.append(float(episode_reward))
        architecture_history.append(list(arch))

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
        architecture_history,
        q_loss_history,
        policy_loss_history,
        total_loss_history
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
        architecture_history,
        q_loss_history,
        policy_loss_history,
        total_loss_history
    ) = train_softq_nas()

    print()
    print("BEST ARCH:", best_arch)
    print("BEST REWARD:", best_reward)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------
    # Save JSON log FIRST
    # -----------------------------
    run_data = {
        "best_arch": list(best_arch),
        "best_reward": float(best_reward),

        "reward_history": [float(x) for x in reward_history],

        "q_loss_history": [float(x) for x in q_loss_history],
        "policy_loss_history": [float(x) for x in policy_loss_history],
        "total_loss_history": [float(x) for x in total_loss_history],

        "architecture_history": architecture_history,
    }

    save_path = os.path.join(
        RESULTS_DIR,
        f"softq_policy_{np.random.randint(1e9)}.json"
    )

    with open(save_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"Saved results to: {save_path}")

    # -----------------------------
    # Plots AFTER logging
    # -----------------------------
    save_reward_plot(reward_history, RESULTS_DIR)

    save_loss_plot(
        q_loss_history,
        policy_loss_history,
        RESULTS_DIR
    )

    save_architecture_heatmap(
        architecture_history,
        RESULTS_DIR
    )

    save_best_arch(
        best_arch,
        best_reward,
        RESULTS_DIR
    )

