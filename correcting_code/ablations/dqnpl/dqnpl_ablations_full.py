import os
import json
import random
import numpy as np
import tensorflow as tf
from collections import deque
from nats_bench import create
from tensorflow.keras import Model, layers

# =========================================================
# GLOBAL CONFIG
# =========================================================

NUM_EDGES = 6
NUM_OPS = 5
STATE_SIZE = NUM_EDGES + 1

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
EPISODES = 200

ALPHA = 1.0
RESULTS_DIR = "nas_ablation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# EXPERIMENT SET (WORKS FOR BOTH OF YOUR LISTS)
# =========================================================

EXPERIMENTS = []  # <-- REPLACE WITH EITHER SET

EXPERIMENTS = [
    {
        "name": "E0_baseline",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E2_gumbel",
        "exploration": "softmax",
        "action_sampler": "gumbel",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E4_prioritized_replay",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "prioritized",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E5_positional_state",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "positional",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E6_flops_reward",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "acc_flops",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E7_double_dqn",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": True,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E8_dueling",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": True,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    {
        "name": "E9_cifar100",
        "exploration": "softmax",
        "action_sampler": "softmax",
        "replay_type": "uniform",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar100",
        "entropy_reg": False,
    },
    {
        "name": "E10_random",
        "exploration": "random",
        "action_sampler": "random",
        "replay_type": "none",
        "state_encoding": "raw",
        "reward_type": "accuracy",
        "double_dqn": False,
        "dueling": False,
        "dataset": "cifar10",
        "entropy_reg": False,
    },
    ##########################rl stability experiments##########################
    {
        "name": "FULL_BASELINE",
        "entropy_reg": False,
        "entropy_beta": 0.0,
        "replay_type": "uniform",
        "double_dqn": False,
        "reward_type": "accuracy",
    },
    {
        "name": "ENTROPY_ON",
        "entropy_reg": True,
        "entropy_beta": 0.01,
        "replay_type": "uniform",
        "double_dqn": False,
        "reward_type": "accuracy",
    },
    {
        "name": "PRIORITIZED_REPLAY",
        "entropy_reg": False,
        "entropy_beta": 0.0,
        "replay_type": "prioritized",
        "double_dqn": False,
        "reward_type": "accuracy",
    },
    {
        "name": "DOUBLE_DQN",
        "entropy_reg": False,
        "entropy_beta": 0.0,
        "replay_type": "uniform",
        "double_dqn": True,
        "reward_type": "accuracy",
    },
    {
        "name": "PARETO_FLOPS",
        "entropy_reg": False,
        "entropy_beta": 0.0,
        "replay_type": "uniform",
        "double_dqn": False,
        "reward_type": "acc_flops",
    },
]


# =========================================================
# STATE ENCODING (RAW + POSITIONAL)
# =========================================================

def encode_state(edges, mode="raw"):
    if mode == "raw":
        return np.concatenate([edges, [0]])

    if mode == "positional":
        pos = np.linspace(0, 1, NUM_EDGES)
        return np.concatenate([edges, pos, [0]])

    return np.concatenate([edges, [0]])

# =========================================================
# ARCH UTILITY
# =========================================================

def state_to_arch(edges):
    return tuple(int(max(0, x)) for x in edges)

# =========================================================
# ENVIRONMENT
# =========================================================

class NATSNASEnv:
    def __init__(self, api, cfg):
        self.api = api
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.edges = np.full(NUM_EDGES, -1, dtype=np.float32)
        return encode_state(self.edges, self.cfg["state_encoding"])

    def step(self, action):
        pos, op = action
        self.edges[int(pos)] = op

        done = np.all(self.edges >= 0)
        reward = self.evaluate() if done else 0.0

        return encode_state(self.edges, self.cfg["state_encoding"]), reward, done, {}

    def evaluate(self):
        arch = state_to_arch(self.edges)
        idx = self.api.query_index_by_arch(arch)

        if idx < 0:
            return 0.0

        info = self.api.get_more_info(
            idx,
            dataset=self.cfg["dataset"],
            hp="200",
            is_random=False
        )

        acc = float(info["valid-accuracy"]) / 100.0

        if self.cfg["reward_type"] == "accuracy":
            return acc

        if self.cfg["reward_type"] == "acc_flops":
            flops = sum(arch)
            return acc - 1e-9 * flops

        return acc

# =========================================================
# MODEL (DUELING OPTIONAL)
# =========================================================

def create_model(cfg):
    inp = layers.Input(shape=(STATE_SIZE if cfg["state_encoding"] == "raw" else STATE_SIZE + NUM_EDGES,))

    x = layers.Dense(512, activation="relu")(inp)
    x = layers.Dense(512, activation="relu")(x)

    if cfg["dueling"]:
        v = layers.Dense(256, activation="relu")(x)
        v = layers.Dense(1)(v)

        a = layers.Dense(256, activation="relu")(x)
        a = layers.Dense(NUM_EDGES * NUM_OPS)(a)
        a = layers.Reshape((NUM_EDGES, NUM_OPS))(a)

        q_pos = tf.reduce_mean(a, axis=2)
        q_op = tf.reduce_mean(a, axis=1)
    else:
        q_pos = layers.Dense(NUM_EDGES)(x)
        q_op = layers.Dense(NUM_OPS)(x)

    return Model(inp, [q_pos, q_op])

# =========================================================
# ACTION SAMPLING (ALL STRATEGIES UNIFIED)
# =========================================================

def select_action(pos_q, op_q, cfg):

    if cfg["action_sampler"] == "random":
        return (
            np.random.randint(NUM_EDGES),
            np.random.randint(NUM_OPS)
        )

    if cfg["action_sampler"] == "gumbel":
        pos = tf.random.categorical(pos_q, 1)[0, 0].numpy()
        op = tf.random.categorical(op_q, 1)[0, 0].numpy()
        return pos, op

    # softmax default
    pos_probs = tf.nn.softmax(pos_q[0] / ALPHA).numpy()
    op_probs = tf.nn.softmax(op_q[0] / ALPHA).numpy()

    return (
        np.random.choice(NUM_EDGES, p=pos_probs),
        np.random.choice(NUM_OPS, p=op_probs)
    )

# =========================================================
# DOUBLE DQN TARGET (OPTIONAL)
# =========================================================

def double_dqn_target(model, target, next_states, rewards, dones):

    next_pos_q, next_op_q = model(next_states, training=False)

    next_pos = tf.argmax(next_pos_q, axis=1)
    next_op = tf.argmax(next_op_q, axis=1)

    t_pos_q, t_op_q = target(next_states, training=False)

    q_pos = tf.reduce_sum(t_pos_q * tf.one_hot(next_pos, NUM_EDGES), axis=1)
    q_op = tf.reduce_sum(t_op_q * tf.one_hot(next_op, NUM_OPS), axis=1)

    return rewards + GAMMA * (q_pos + q_op) * (1.0 - dones)

# =========================================================
# ENTROPY REGULARIZATION
# =========================================================

def entropy(q):
    p = tf.nn.softmax(q / ALPHA)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=1))

# =========================================================
# REPLAY BUFFER (UNIFORM + PRIORITIZED)
# =========================================================

class PrioritizedReplay:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, transition, td_error=1.0):
        p = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            idx = np.argmin(self.priorities)
            self.buffer[idx] = transition
            self.priorities[idx] = p

    def sample(self, batch_size):
        probs = np.array(self.priorities)
        probs = probs / probs.sum()

        idx = np.random.choice(len(self.buffer), batch_size, p=probs)

        batch = [self.buffer[i] for i in idx]
        weights = (len(self.buffer) * probs[idx]) ** (-1)
        weights /= weights.max()

        return batch, weights, idx

# =========================================================
# TRAIN LOOP (FULL ABSTRACTION)
# =========================================================

def run_experiment(cfg):

    api = create(
        "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
        "tss",
        fast_mode=True
    )

    env = NATSNASEnv(api, cfg)

    model = create_model(cfg)
    target = create_model(cfg)
    target.set_weights(model.get_weights())

    opt = tf.keras.optimizers.Adam(LR)

    replay = (
        PrioritizedReplay() if cfg["replay_type"] == "prioritized"
        else deque(maxlen=50000)
    )

    for ep in range(EPISODES):

        state = env.reset()
        done = False
        ep_reward = 0

        while not done:

            s = state[None].astype(np.float32)

            pos_q, op_q = model(s, training=False)

            action = select_action(pos_q, op_q, cfg)

            next_state, reward, done, _ = env.step(action)

            if cfg["replay_type"] == "prioritized":
                replay.add((state, action, reward, next_state, done))
            else:
                replay.append((state, action, reward, next_state, done))

            state = next_state
            ep_reward += reward

            if len(replay) >= BATCH_SIZE:

                if cfg["replay_type"] == "prioritized":
                    batch, weights, idxs = replay.sample(BATCH_SIZE)
                else:
                    batch = random.sample(replay, BATCH_SIZE)
                    weights = np.ones(BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states, np.float32)
                next_states = np.array(next_states, np.float32)
                actions = np.array(actions)
                rewards = np.array(rewards, np.float32)
                dones = np.array(dones, np.float32)

                with tf.GradientTape() as tape:

                    q_pos, q_op = model(states)

                    a1 = tf.one_hot(actions[:, 0], NUM_EDGES)
                    a2 = tf.one_hot(actions[:, 1], NUM_OPS)

                    q_taken = tf.reduce_sum(q_pos * a1, axis=1) + \
                              tf.reduce_sum(q_op * a2, axis=1)

                    if cfg["double_dqn"]:
                        target_q = double_dqn_target(
                            model, target, next_states, rewards, dones
                        )
                    else:
                        nq_pos, nq_op = target(next_states, training=False)
                        target_q = rewards + GAMMA * (
                            tf.reduce_max(nq_pos, axis=1) +
                            tf.reduce_max(nq_op, axis=1)
                        ) * (1 - dones)

                    td_error = target_q - q_taken

                    loss = tf.reduce_mean(weights * tf.square(td_error))

                    if cfg.get("entropy_reg", False):
                        loss -= cfg.get("entropy_beta", 0.01) * (
                            entropy(q_pos) + entropy(q_op)
                        )

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

                if cfg["replay_type"] == "prioritized":
                    for i, idx in enumerate(idxs):
                        replay.priorities[idx] = abs(float(td_error[i].numpy()))

        if ep % 10 == 0:
            target.set_weights(model.get_weights())

        print(cfg["name"], ep, ep_reward)

    return ep_reward

# =========================================================
# GRID RUNNER
# =========================================================

if __name__ == "__main__":

    results = {}

    for cfg in EXPERIMENTS:
        print("\nRUN:", cfg["name"])
        results[cfg["name"]] = run_experiment(cfg)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE:", results)