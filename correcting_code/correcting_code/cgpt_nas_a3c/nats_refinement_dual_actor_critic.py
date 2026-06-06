"""
NATS-Bench Sequence Refinement NAS
Dual Actor-Critic + VQ-VAE Policies

State:
[edge01, edge02, edge12, edge03, edge13, edge23,
 accuracy_bucket, target_improvement]

Action:
(position_to_edit, operation_to_write)


state = [
    edge01,
    edge02,
    edge12,
    edge03,
    edge13,
    edge23,
    accuracy_bucket,
    target_improvement,
]
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
from gymnasium import spaces
from nats_bench import create

import os
import json
import matplotlib.pyplot as plt

# ======================================================
# CONSTANTS
# ======================================================

NUM_EDGES = 6
NUM_OPS = 4

STATE_LEN = 8

NUM_HIDDEN = 128
NUM_EMBEDDINGS = 64

GAMMA = 0.99

MAX_EPISODE_STEPS = 10

OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
]
#Plot utilities

def save_accuracy_plot(acc_history, output_dir):

    plt.figure(figsize=(10, 6))

    plt.plot(acc_history)

    plt.xlabel("Episode")
    plt.ylabel("Accuracy")

    plt.title("NATS-Bench Accuracy Evolution")

    plt.grid(True)

    plt.savefig(
        os.path.join(
            output_dir,
            "accuracy_curve.png"
        )
    )

    plt.close()

def save_loss_plot(
    action_losses,
    position_losses,
    output_dir,
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        action_losses,
        label="Action Policy"
    )

    plt.plot(
        position_losses,
        label="Position Policy"
    )

    plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.title(
        "Dual Actor-Critic Loss"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            output_dir,
            "loss_curve.png"
        )
    )

    plt.close()


def save_architecture_heatmap(
    architectures,
    output_dir,
):

    arr = np.array(
        architectures,
        dtype=np.int32
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        arr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xlabel("Edge")

    plt.ylabel("Episode")

    plt.title(
        "Architecture Evolution"
    )

    plt.savefig(
        os.path.join(
            output_dir,
            "architecture_evolution.png"
        )
    )

    plt.close()


def save_best_architecture(
    best_arch,
    best_acc,
    output_dir,
):

    plt.figure(
        figsize=(10, 2)
    )

    plt.bar(
        np.arange(
            len(best_arch)
        ),
        best_arch
    )

    plt.xlabel("Edge")
    plt.ylabel("Operation")

    plt.title(
        f"Best Accuracy={best_acc:.4f}"
    )

    plt.savefig(
        os.path.join(
            output_dir,
            "best_architecture.png"
        )
    )

    plt.close()





# ======================================================
# VQ-VAE
# ======================================================

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        w_init = tf.random_uniform_initializer()

        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(embedding_dim, num_embeddings),
                dtype="float32",
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def get_code_indices(self, flattened_inputs):

        similarity = tf.matmul(
            flattened_inputs,
            self.embeddings
        )

        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        return tf.argmin(distances, axis=1)

    def call(self, x):

        input_shape = tf.shape(x)

        flattened = tf.reshape(
            x,
            [-1, self.embedding_dim]
        )

        encoding_indices = self.get_code_indices(flattened)

        encodings = tf.one_hot(
            encoding_indices,
            self.num_embeddings
        )

        quantized = tf.matmul(
            encodings,
            self.embeddings,
            transpose_b=True
        )

        quantized = tf.reshape(
            quantized,
            input_shape
        )

        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )

        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(x)) ** 2
        )

        self.add_loss(commitment_loss + codebook_loss)

        return x + tf.stop_gradient(quantized - x)


def get_encoder(inputs,
                head_size=256,
                num_heads=4,
                ff_dim=4,
                dropout=0.25):

    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)

    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    res = x + inputs

    x = layers.Conv1D(
        filters=ff_dim,
        kernel_size=1,
        activation="relu"
    )(res)

    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(
        filters=inputs.shape[-1],
        kernel_size=1
    )(x)

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    encoder_outputs = x + res

    return encoder_outputs


def build_encoder_model(
    input_shape=(STATE_LEN,),
    latent_dim=NUM_HIDDEN,
    blocks=4,
):

    seq_len = input_shape[0]

    inputs = keras.Input(shape=input_shape)

    x = layers.Reshape((seq_len, 1))(inputs)

    x = layers.Conv1D(
        latent_dim,
        1,
        padding="same"
    )(x)

    for _ in range(blocks):
        x = get_encoder(x)

    return keras.Model(
        inputs,
        x,
        name="encoder"
    )


def get_decoder(seq_len, latent_dim):

    inputs = keras.Input(
        shape=(seq_len, latent_dim)
    )

    x = layers.Flatten()(inputs)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        latent_dim,
        activation="relu"
    )(x)

    return keras.Model(inputs, outputs)


def get_vqvae(output_dim):

    encoder = build_encoder_model(
        input_shape=(STATE_LEN,),
        latent_dim=NUM_HIDDEN
    )

    decoder = get_decoder(
        STATE_LEN,
        NUM_HIDDEN
    )

    vq = VectorQuantizer(
        NUM_EMBEDDINGS,
        NUM_HIDDEN
    )

    inputs = keras.Input(
        shape=(STATE_LEN,)
    )

    x = encoder(inputs)
    x = vq(x)
    x = decoder(x)

    actor = layers.Dense(
        output_dim,
        activation="softmax",
        name="actor"
    )(x)

    critic = layers.Dense(
        1,
        name="critic"
    )(x)

    return keras.Model(
        inputs,
        [actor, critic]
    )

# ======================================================
# NATS HELPERS
# ======================================================

def encoding_to_arch(tokens):

    e = [OPS[int(x)] for x in tokens]

    return (
        f"|{e[0]}~0|+"
        f"|{e[1]}~0|{e[2]}~1|+"
        f"|{e[3]}~0|{e[4]}~1|{e[5]}~2|"
    )

# ======================================================
# ENV
# ======================================================

class NATSRefinementEnv(gym.Env):

    def __init__(self, api):

        self.api = api

        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(NUM_EDGES),
                spaces.Discrete(NUM_OPS),
            )
        )

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(STATE_LEN,),
            dtype=np.int32,
        )

    def evaluate_architecture(self, arch):

        try:

            arch_str = encoding_to_arch(arch)

            idx = self.api.query_index_by_arch(
                arch_str
            )

            info = self.api.get_more_info(
                idx,
                dataset="cifar10",
                hp="200",
                is_random=False,
            )

            return float(
                info.get(
                    "valid-accuracy",
                    info.get("test-accuracy", 0.0)
                )
            )

        except Exception:
            return 0.0

    def make_state(self):

        acc_bucket = int(
            self.current_accuracy // 2
        )

        return np.concatenate(
            [
                self.architecture,
                np.array(
                    [
                        acc_bucket,
                        self.target_improvement
                    ],
                    dtype=np.int32
                )
            ]
        )

    def reset(self, seed=None, options=None):

        self.target_improvement = 1

        self.architecture = np.array(
            [3, 3, 3, 3, 3, 3],
            dtype=np.int32
        )

        self.current_accuracy = self.evaluate_architecture(
            self.architecture
        )

        self.steps = 0

        return self.make_state(), {}

    def step(self, action):

        position, op = action

        old_acc = self.current_accuracy

        self.architecture[position] = op

        self.current_accuracy = self.evaluate_architecture(
            self.architecture
        )

        reward = (
            self.current_accuracy - old_acc
        ) * 10.0

        self.steps += 1

        done = self.steps >= MAX_EPISODE_STEPS

        return (
            self.make_state(),
            reward,
            done,
            False,
            {}
        )

# ======================================================
# TRAINING
# ======================================================

def train_nas_agent(env, episodes=100):

    model_action = get_vqvae(NUM_OPS)
    model_pos = get_vqvae(NUM_EDGES)

    opt_action = keras.optimizers.Adam(1e-4)
    opt_pos = keras.optimizers.Adam(1e-4)

    best_acc = -1
    best_arch = None

    accuracy_history = []

    action_loss_history = []
    position_loss_history = []

    architecture_history = []

    for ep in range(episodes):

        state, _ = env.reset()

        rewards = []

        a_logs = []
        p_logs = []

        a_vals = []
        p_vals = []

        with tf.GradientTape() as tape_a, \
             tf.GradientTape() as tape_p:

            done = False

            while not done:

                s = tf.expand_dims(
                    tf.convert_to_tensor(
                        state,
                        dtype=tf.float32
                    ),
                    0
                )

                action_probs, action_value = model_action(
                    s,
                    training=True
                )

                pos_probs, pos_value = model_pos(
                    s,
                    training=True
                )

                action_probs = tf.squeeze(action_probs)
                pos_probs = tf.squeeze(pos_probs)

                action = np.random.choice(
                    NUM_OPS,
                    p=action_probs.numpy()
                )

                position = np.random.choice(
                    NUM_EDGES,
                    p=pos_probs.numpy()
                )

                a_logs.append(
                    tf.math.log(
                        action_probs[action] + 1e-8
                    )
                )

                p_logs.append(
                    tf.math.log(
                        pos_probs[position] + 1e-8
                    )
                )

                a_vals.append(tf.squeeze(action_value))
                p_vals.append(tf.squeeze(pos_value))

                state, reward, done, _, _ = env.step(
                    (position, action)
                )

                rewards.append(reward)

            returns = []
            d = 0.0

            for r in reversed(rewards):
                d = r + GAMMA * d
                returns.insert(0, d)

            returns = tf.convert_to_tensor(
                returns,
                dtype=tf.float32
            )

            if len(returns) > 1:
                returns = (
                    returns - tf.reduce_mean(returns)
                ) / (
                    tf.math.reduce_std(returns) + 1e-8
                )

            loss_a = 0.0
            loss_p = 0.0

            for lp, v, ret in zip(
                a_logs,
                a_vals,
                returns
            ):

                adv = ret - v

                loss_a += (
                    -lp * tf.stop_gradient(adv)
                    + tf.square(adv)
                )

            for lp, v, ret in zip(
                p_logs,
                p_vals,
                returns
            ):

                adv = ret - v

                loss_p += (
                    -lp * tf.stop_gradient(adv)
                    + tf.square(adv)
                )

        grads = tape_a.gradient(
            loss_a,
            model_action.trainable_variables
        )

        opt_action.apply_gradients(
            zip(
                grads,
                model_action.trainable_variables
            )
        )

        grads = tape_p.gradient(
            loss_p,
            model_pos.trainable_variables
        )

        opt_pos.apply_gradients(
            zip(
                grads,
                model_pos.trainable_variables
            )
        )

        action_loss_history.append(
            float(loss_a.numpy())
        )

        position_loss_history.append(
            float(loss_p.numpy())
        )

        if env.current_accuracy > best_acc:
            best_acc = env.current_accuracy
            best_arch = env.architecture.copy()

        print(
            f"Episode={ep+1} "
            f"Acc={env.current_accuracy:.3f} "
            f"Best={best_acc:.3f}"
        )

        accuracy_history.append(
            float(env.current_accuracy)
        )

        architecture_history.append(
            env.architecture.copy()
        )


    print("\\nBest Architecture:", best_arch)
    print("Best Accuracy:", best_acc)

    return (
        model_action,
        model_pos,
        best_arch,
        best_acc,
        accuracy_history,
        action_loss_history,
        position_loss_history,
        architecture_history,
    )

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    os.makedirs(
        "results_refinement",
        exist_ok=True
    )

    api = create(
        "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSRefinementEnv(api)

    (
        model_action,
        model_pos,
        best_arch,
        best_acc,
        accuracy_history,
        action_loss_history,
        position_loss_history,
        architecture_history,
    ) = train_nas_agent(
        env,
        episodes=100
    )

    save_accuracy_plot(
        accuracy_history,
        "results_refinement"
    )

    save_loss_plot(
        action_loss_history,
        position_loss_history,
        "results_refinement"
    )

    save_architecture_heatmap(
        architecture_history,
        "results_refinement"
    )

    save_best_architecture(
        best_arch,
        best_acc,
        "results_refinement"
    )

    run_data = {
        "best_arch": best_arch.tolist(),
        "best_accuracy": float(best_acc),
        "accuracy_history": [
            float(x)
            for x in accuracy_history
        ],
        "action_loss_history": [
            float(x)
            for x in action_loss_history
        ],
        "position_loss_history": [
            float(x)
            for x in position_loss_history
        ],
        "best_arch_string": encoding_to_arch(best_arch),
    }

    run_path = os.path.join(
        "results_refinement",
        f"nats_refinement_run_{np.random.randint(0,1e9)}.json"
    )

    with open(run_path, "w") as f:
        json.dump(
            run_data,
            f,
            indent=2
        )

    print(
        f"Saved run to {run_path}"
    )



"""
results_refinement/
│
├── accuracy_curve.png
├── loss_curve.png
├── architecture_evolution.png
├── best_architecture.png
│
└── nats_refinement_run_84729311.json

"""