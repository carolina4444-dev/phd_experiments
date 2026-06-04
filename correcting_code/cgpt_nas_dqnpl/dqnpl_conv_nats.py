import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from nats_bench import create
import os, json
import matplotlib.pyplot as plt

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
    ops = ["none", "skip_connect", "conv_1x1", "conv_3x3"]

    arch = []
    for op in tree_encoding:
        if op == 0:
            arch.append(1)
        elif op == 1:
            arch.append(2)
        else:
            arch.append(3)

    arch = arch[:6]
    arch += [1] * (6 - len(arch))

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

            if len(self.tree_encoding) >= 6:  # IMPORTANT: NATS-TSS = 6 edges
                done = True

        reward = 0.0

        if done:
            reward = self.evaluate()

        return self.state.copy(), reward, done, {}

    def evaluate(self):
        arch = encoding_to_nats_arch(self.tree_encoding)

        info = self.api.get_more_info(
            arch,
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

    # position selection (which node to edit)
    pos_q = layers.Dense(MAX_NODES, name="pos_q")(x)

    # value selection (0,1,STOP)
    val_q = layers.Dense(NUM_ACTIONS, name="val_q")(x)

    return tf.keras.Model(inp, [pos_q, val_q])


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

            pos_q, val_q = model(s, training=False)

            pos_probs = tf.nn.softmax(pos_q[0] / ALPHA).numpy()
            val_probs = tf.nn.softmax(val_q[0] / ALPHA).numpy()

            position = np.random.choice(MAX_NODES, p=pos_probs)
            value = np.random.choice(NUM_ACTIONS, p=val_probs)

            next_state, reward, done, _ = env.step((position, value))

            replay.add((state, position, value, reward, next_state, done))

            state = next_state
            total_reward += reward

            # =====================================================
            # TRAIN STEP
            # =====================================================
            if len(replay) >= batch_size:

                batch = replay.sample(batch_size)

                states, pos, val, rewards, next_states, dones = zip(*batch)

                states = np.array(states, dtype=np.float32)
                next_states = np.array(next_states, dtype=np.float32)
                rewards = np.array(rewards, dtype=np.float32)
                dones = np.array(dones, dtype=np.float32)

                pos = np.array(pos)
                val = np.array(val)

                with tf.GradientTape() as tape:

                    pos_q, val_q = model(states, training=True)

                    # entropy
                    pos_p = tf.nn.softmax(pos_q / ALPHA)
                    val_p = tf.nn.softmax(val_q / ALPHA)

                    entropy = tf.reduce_mean(
                        -tf.reduce_sum(pos_p * tf.math.log(pos_p + 1e-8), axis=1)
                        -tf.reduce_sum(val_p * tf.math.log(val_p + 1e-8), axis=1)
                    )

                    npos_q, nval_q = target(next_states, training=False)

                    next_v = soft_value(npos_q, ALPHA) + soft_value(nval_q, ALPHA)

                    target_q = rewards + GAMMA * next_v * (1.0 - dones)

                    q_pos = tf.reduce_sum(pos_q * tf.one_hot(pos, MAX_NODES), axis=1)
                    q_val = tf.reduce_sum(val_q * tf.one_hot(val, NUM_ACTIONS), axis=1)

                    pred_q = q_pos + q_val

                    loss_q = tf.reduce_mean(tf.square(target_q - pred_q))

                    loss = loss_q - 0.01 * entropy

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

            if done:
                break

        if ep % 10 == 0:
            target.set_weights(model.get_weights())

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
    "results",
    f"nats_softq_run_{int(np.random.randint(0, 1e9))}.json"
)

    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"Saved run to {run_path}")