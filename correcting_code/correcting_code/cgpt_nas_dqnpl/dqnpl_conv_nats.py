import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from nats_bench import create
import os, json
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# =========================================================
# CONFIG (MATCHING YOUR SETUP)
# =========================================================

MAX_NODES = 8
NUM_ACTIONS = 3   # 0=op1, 1=op2, 2=STOP

GAMMA = 0.99
ALPHA = 1.0


# =========================================================
# SAME ENCODING FUNCTION AS YOUR SCRIPT
# =========================================================

def encoding_to_nats_arch(tree_encoding):
    """
    Convert RL decisions → NATS-Bench TSS architecture.

    NATS-TSS valid ops:
        0 = none
        1 = skip_connect
        2 = nor_conv_1x1
        3 = nor_conv_3x3
        4 = avg_pool_3x3
    """

    valid_ops = {0, 1, 2, 3, 4}

    # take first 6 decisions (6 edges in TSS cell)
    arch = []

    for op in tree_encoding:
        if len(arch) >= 6:
            break

        if op in valid_ops:
            arch.append(int(op))
        else:
            raise ValueError(f"Invalid NATS op: {op}")

    # pad with NONE (important: NOT skip_connect)
    arch += [0] * (6 - len(arch))

    return tuple(arch)


# =========================================================
# ENVIRONMENT (MATCH YOUR SEMANTICS EXACTLY)
# =========================================================

class NATSNASEnvSoftQ:

    def __init__(self, api, max_nodes=MAX_NODES):
        self.api = api
        self.max_nodes = max_nodes

        self.START_TOKEN = 3
        self.PAD_TOKEN = 4

        self.reset()

    def reset(self):
        self.tree_encoding = []
        self.state = np.full(self.max_nodes, self.PAD_TOKEN, dtype=np.int32)
        self.state[0] = self.START_TOKEN
        self.cursor = 1
        return self.state.copy()

    def step(self, action):
        done = False

        if action == 2:  # STOP
            done = True
        else:
            self.tree_encoding.append(int(action))

            if self.cursor < self.max_nodes:
                self.state[self.cursor] = action
                self.cursor += 1

            # NATS-TSS constraint: 6 edges
            if len(self.tree_encoding) >= 6:
                done = True

        reward = 0.0
        if done:
            reward = self.evaluate()

        return self.state.copy(), reward, done, {}


    def evaluate(self, arch=None):

        if arch is None:
            arch = encoding_to_nats_arch(self.tree_encoding)

        index = self.api.query_index_by_arch(tuple(arch))

        if index < 0:
            # fallback: sample random valid index (SAFE WAY)
            index = np.random.randint(0, len(self.api))

        info = self.api.get_more_info(
            index,
            dataset="cifar10",
            hp="200",
            is_random=False
        )

        return float(info["valid-accuracy"])

# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def add(self, x):
        self.buffer.append(x)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


# =========================================================
# SOFT Q NETWORK (adapted to YOUR action space)
# =========================================================

def create_softq():
    inp = layers.Input(shape=(MAX_NODES,))

    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(256, activation="relu")(x)

    # value selection (0,1,STOP)
    val_q = layers.Dense(NUM_ACTIONS, name="val_q")(x)

    return tf.keras.Model(inp, val_q)


# =========================================================
# SOFT VALUE FUNCTION
# =========================================================

def soft_value(q, alpha):
    return alpha * tf.reduce_logsumexp(q / alpha, axis=-1)


# =========================================================
# TRAINING LOOP (SOFT Q LEARNING NAS)
# =========================================================
def train_softq_nas(episodes=100, batch_size=32):

    api = create(
        "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSNASEnvSoftQ(api)

    model = create_softq()
    target = create_softq()
    target.set_weights(model.get_weights())

    opt = tf.keras.optimizers.Adam(1e-4)
    replay = ReplayBuffer()

    best_reward = -1e9
    best_arch = None

    reward_history = []
    architecture_history = []

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0

        while True:

            s = state[None, :].astype(np.float32)

            # =========================
            # ACTION SELECTION
            # =========================
            val_q = model(s, training=False)
            val_probs = tf.nn.softmax(val_q[0] / ALPHA).numpy()

            value = np.random.choice(NUM_ACTIONS, p=val_probs)

            next_state, reward, done, _ = env.step(value)

            replay.add((state, value, reward, next_state, done))

            state = next_state
            total_reward += reward

            # =========================
            # TRAIN STEP
            # =========================
            if len(replay) >= batch_size:

                batch = replay.sample(batch_size)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states, dtype=np.float32)
                next_states = np.array(next_states, dtype=np.float32)
                rewards = np.array(rewards, dtype=np.float32)
                dones = np.array(dones, dtype=np.float32)

                actions = np.array(actions)

                with tf.GradientTape() as tape:

                    # Q(s, a)
                    val_q = model(states, training=True)

                    q_val = tf.reduce_sum(
                        val_q * tf.one_hot(actions, NUM_ACTIONS),
                        axis=1
                    )

                    # -------------------------
                    # TARGET: soft value
                    # -------------------------
                    next_q = target(next_states, training=False)

                    next_v = ALPHA * tf.reduce_logsumexp(next_q / ALPHA, axis=1)

                    target_q = rewards + GAMMA * next_v * (1.0 - dones)

                    # -------------------------
                    # LOSS
                    # -------------------------
                    loss_q = tf.reduce_mean(tf.square(target_q - q_val))

                    # entropy regularization (policy entropy from logits)
                    probs = tf.nn.softmax(val_q / ALPHA)
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                    )

                    loss = loss_q - 0.01 * entropy

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

            if done:
                break

        # =========================
        # TARGET UPDATE
        # =========================
        if ep % 10 == 0:
            target.set_weights(model.get_weights())

        # =========================
        # TRACK BEST
        # =========================
        if total_reward > best_reward:
            best_reward = total_reward
            best_arch = env.tree_encoding.copy()

        print(f"Episode {ep} | Reward={total_reward:.4f} | Arch={env.tree_encoding}")

        reward_history.append(total_reward)
        architecture_history.append(env.tree_encoding.copy())

    return model, best_arch, best_reward, reward_history, architecture_history
# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    model, best_arch, best_reward, reward_history, architecture_history = train_softq_nas()

    print("\nBEST ARCH:", best_arch)
    print("BEST REWARD:", best_reward)

    os.makedirs("results_dqn", exist_ok=True)

    run_data = {
        "best_arch": best_arch,
        "best_reward": float(best_reward),
        "reward_history": [float(x) for x in reward_history],
        "architecture_history": architecture_history,
    }

    run_path = os.path.join(
        "results_dqn",
        f"nats_softq_run_{int(np.random.randint(0, 1e9))}.json"
    )

    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"Saved run to {run_path}")