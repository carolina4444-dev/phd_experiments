

# | ID  | Exploration | Action  | Replay      | State      | Reward    | DQN        | Dataset  | Notes                     |
# | --- | ----------- | ------- | ----------- | ---------- | --------- | ---------- | -------- | ------------------------- |
# | E0  | softmax     | softmax | uniform     | raw        | acc       | vanilla    | CIFAR10  | baseline                  |
# | E1  | epsilon     | softmax | uniform     | raw        | acc       | vanilla    | CIFAR10  | epsilon greedy            |
# | E2  | softmax     | gumbel  | uniform     | raw        | acc       | vanilla    | CIFAR10  | stochastic differentiable |
# | E3  | softmax     | argmax  | uniform     | raw        | acc       | vanilla    | CIFAR10  | deterministic policy      |
# | E4  | softmax     | softmax | prioritized | raw        | acc       | vanilla    | CIFAR10  | replay ablation           |
# | E5  | softmax     | softmax | uniform     | positional | acc       | vanilla    | CIFAR10  | state encoding            |
# | E6  | softmax     | softmax | uniform     | raw        | acc+flops | vanilla    | CIFAR10  | reward shaping            |
# | E7  | softmax     | softmax | uniform     | raw        | acc       | double DQN | CIFAR10  | stability                 |
# | E8  | softmax     | softmax | uniform     | raw        | acc       | dueling    | CIFAR10  | value decomposition       |
# | E9  | softmax     | softmax | uniform     | raw        | acc       | vanilla    | CIFAR100 | transfer                  |
# | E10 | random      | random  | none        | raw        | acc       | none       | CIFAR10  | random search baseline    |



import os
import json
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from nats_bench import create
import matplotlib.pyplot as plt

# =========================================================
# EXPERIMENT GRID
# =========================================================

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
]

# =========================================================
# CONFIG BASE
# =========================================================

NUM_EDGES = 6
NUM_OPS = 5
STATE_SIZE = NUM_EDGES + 1

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
EPISODES = 200
ALPHA = 1.0

RESULTS_DIR = "grid_results_nas"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# ENV
# =========================================================

def state_to_arch(edges):
    return tuple(int(max(0, x)) for x in edges)


class NATSNASEnv:
    def __init__(self, api, config):
        self.api = api
        self.config = config
        self.reset()

    def reset(self):
        self.edges = np.full(NUM_EDGES, -1, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.edges, [0]])

    def step(self, action):
        pos, op = action
        self.edges[int(pos)] = op
        done = np.all(self.edges >= 0)
        reward = self.evaluate() if done else 0.0
        return self._get_state(), reward, done, {}

    def evaluate(self):
        arch = state_to_arch(self.edges)
        idx = self.api.query_index_by_arch(arch)
        if idx < 0:
            return 0.0

        info = self.api.get_more_info(
            idx,
            dataset=self.config["dataset"],
            hp="200",
            is_random=False
        )

        acc = float(info["valid-accuracy"]) / 100.0

        if self.config["reward_type"] == "acc_flops":
            flops_penalty = 1e-9 * sum(arch)
            return acc - flops_penalty

        return acc

# =========================================================
# MODEL (with optional dueling)
# =========================================================

def create_model(cfg):
    inp = layers.Input(shape=(STATE_SIZE,))
    x = layers.Dense(512)(inp)
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
# ACTION SELECTION ABSTRACTION
# =========================================================

def sample_action(pos_q, op_q, cfg):
    if cfg["action_sampler"] == "random":
        return np.random.randint(NUM_EDGES), np.random.randint(NUM_OPS)

    if cfg["action_sampler"] == "gumbel":
        pos = tf.random.categorical(pos_q, 1)[0,0].numpy()
        op = tf.random.categorical(op_q, 1)[0,0].numpy()
        return pos, op

    pos_probs = tf.nn.softmax(pos_q[0] / ALPHA).numpy()
    op_probs = tf.nn.softmax(op_q[0] / ALPHA).numpy()

    return (
        np.random.choice(NUM_EDGES, p=pos_probs),
        np.random.choice(NUM_OPS, p=op_probs)
    )

# =========================================================
# TRAINING LOOP
# =========================================================

def run_experiment(cfg):

    api = create("/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple", "tss", fast_mode=True)

    env = NATSNASEnv(api, cfg)
    model = create_model(cfg)
    target = create_model(cfg)
    target.set_weights(model.get_weights())

    opt = tf.keras.optimizers.Adam(LR)
    replay = deque(maxlen=50000)

    best = -1

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        ep_r = 0

        while not done:
            s = state[None].astype(np.float32)
            pos_q, op_q = target(s, training=False)

            action = sample_action(pos_q, op_q, cfg)
            ns, r, done, _ = env.step(action)

            if cfg["replay_type"] != "none":
                replay.append((state, action, r, ns, done))

            state = ns
            ep_r += r

            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states, np.float32)
                next_states = np.array(next_states, np.float32)
                actions = np.array(actions)
                rewards = np.array(rewards, np.float32)
                dones = np.array(dones, np.float32)

                with tf.GradientTape() as tape:
                    q_pos, q_op = model(states)

                    pos_a = tf.one_hot(actions[:,0], NUM_EDGES)
                    op_a = tf.one_hot(actions[:,1], NUM_OPS)

                    q_taken = tf.reduce_sum(q_pos * pos_a, axis=1) + \
                              tf.reduce_sum(q_op * op_a, axis=1)

                    next_q_pos, next_q_op = target(next_states, training=False)
                    next_v = tf.reduce_max(next_q_pos, axis=1)

                    target_q = rewards + GAMMA * next_v * (1 - dones)

                    loss = tf.reduce_mean((target_q - q_taken) ** 2)

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

        if ep % 10 == 0:
            target.set_weights(model.get_weights())

        best = max(best, ep_r)

        print(cfg["name"], ep, ep_r, best)

    return best

# =========================================================
# GRID RUNNER
# =========================================================

if __name__ == "__main__":

    results = {}

    for cfg in EXPERIMENTS:
        print("\nRUN:", cfg["name"])
        results[cfg["name"]] = run_experiment(cfg)

    with open(os.path.join(RESULTS_DIR, "grid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE:", results)