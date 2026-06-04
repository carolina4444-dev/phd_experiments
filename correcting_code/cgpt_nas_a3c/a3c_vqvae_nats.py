import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from nats_bench import create


import os
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import json

# CONFIGURATION

max_nodes = 8
train_epochs = 1
SHOTS_PER_CLASS = 20

MAX_NODES = max_nodes
NUM_ACTIONS = 3
NUM_HIDDEN = 128

maxlen = 5
num_actions = NUM_ACTIONS
num_hidden = NUM_HIDDEN


from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

from transformer_keras_io import *

def build_tree_model(input_shape, num_classes, tree_encoding, max_length, vocab_size, embedding_dim):
    input_layer = Input(shape=input_shape)
    x = input_layer

    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embedding_dim)
    x = embedding_layer(x)

    for bit in tree_encoding:
        if bit == 0:
            # Node (Inception Module)
            x = inception_module(x)
        elif bit == 1:
            # Leaf (Conv1D Layer)
            x = Conv1D(64, 3, padding='same', activation='relu')(x)
            x = MaxPooling1D(2, strides=2, padding='same')(x)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = Dense(256, activation='relu')(x)

    # Output Layer
    #output_layer = Dense(num_classes, activation='softmax')(x)
    # output_layer = Dense(1, activation='sigmoid')(x)
    output_layer = Dense(4, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def inception_module(x):
    branch1x1 = Conv1D(64, 1, padding='same', activation='relu')(x)

    branch3x3 = Conv1D(128, 1, padding='same', activation='relu')(x)
    branch3x3 = Conv1D(128, 3, padding='same', activation='relu')(branch3x3)

    branch5x5 = Conv1D(32, 1, padding='same', activation='relu')(x)
    branch5x5 = Conv1D(32, 5, padding='same', activation='relu')(branch5x5)

    branch_pool = MaxPooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1D(32, 1, padding='same', activation='relu')(branch_pool)

    return Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])



################################################################################

class NATSNASEnv(gym.Env):
    def __init__(self, api, max_nodes=8):
        super().__init__()

        self.api = api
        self.max_nodes = max_nodes

        self.START_TOKEN = 3
        self.PAD_TOKEN = 4

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0, high=4, shape=(max_nodes,), dtype=np.int32
        )

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

            if len(self.tree_encoding) >= self.max_nodes:
                done = True

        reward = 0.0

        if done:
            reward = self.evaluate_architecture()

        return self.state.copy(), reward, done, {}

    def evaluate_architecture(self):
        arch = encoding_to_nats_arch(self.tree_encoding)

        # NATS query (CIFAR-10 validation accuracy)
        info = self.api.get_more_info(
            arch,
            dataset="cifar10",
            hp="200",   # 12 or 200 epochs benchmark
            is_random=False,
        )

        val_acc = info["valid-accuracy"]

        return float(val_acc)


################################################################################


class NATSBenchProblem:

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        input_shape,
        max_length,
        vocab_size,
        embedding_dim,
        api
    ):

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.input_shape = input_shape

        self.max_length = max_length

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.api = api



##############################################################################
def run_episode(
    env,
    actor_model,
):

    state = env.reset()

    done = False

    episode_reward = 0

    while not done:

        state_tensor = tf.expand_dims(
            tf.convert_to_tensor(state),
            axis=0
        )

        probs, value = actor_model(
            state_tensor,
            training=False
        )

        probs = probs.numpy()[0]

        action = np.random.choice(
            len(probs),
            p=probs
        )

        next_state, reward, done, _ = env.step(
            action
        )

        state = next_state

        episode_reward += reward

    return (
        env.tree_encoding,
        episode_reward
    )


###############################################################################
def get_nas_actor_critic():

    inputs = keras.Input(
        shape=(MAX_NODES,)
    )

    encoder = build_encoder_model(
        input_shape=(MAX_NODES,)
    )

    x = encoder(inputs)

    vq = VectorQuantizer(
        num_embeddings=MAX_NODES,
        embedding_dim=NUM_HIDDEN,
    )(x)

    x = layers.Flatten()(vq)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    actor = layers.Dense(
        NUM_ACTIONS,
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

def train_nas_agent(
    env,
    episodes=100,
):
    reward_history = []
    actor_loss_history = []
    critic_loss_history = []
    architecture_history = []

    gamma = 0.99

    optimizer = keras.optimizers.Adam(
        1e-4
    )

    model = get_nas_actor_critic()

    best_reward = -1e9
    best_architecture = None

    for episode in range(episodes):

        state = env.reset()

        done = False

        action_log_probs = []
        critic_values = []
        rewards = []

        with tf.GradientTape() as tape:

            while not done:

                state_tensor = tf.expand_dims(
                    tf.convert_to_tensor(
                        state,
                        dtype=tf.float32
                    ),
                    axis=0
                )

                action_probs, value = model(
                    state_tensor,
                    training=True
                )

                action_probs = tf.squeeze(
                    action_probs
                )

                action = np.random.choice(
                    3,
                    p=action_probs.numpy()
                )

                log_prob = tf.math.log(
                    action_probs[action]
                    + 1e-8
                )

                next_state, reward, done, _ = env.step(
                    action
                )

                action_log_probs.append(
                    log_prob
                )

                critic_values.append(
                    tf.squeeze(value)
                )

                rewards.append(
                    reward
                )

                state = next_state

            returns = []

            discounted_sum = 0

            for r in reversed(rewards):

                discounted_sum = (
                    r
                    + gamma * discounted_sum
                )

                returns.insert(
                    0,
                    discounted_sum
                )

            returns = tf.convert_to_tensor(
                returns,
                dtype=tf.float32
            )

            returns = (
                returns
                - tf.reduce_mean(
                    returns
                )
            ) / (
                tf.math.reduce_std(
                    returns
                )
                + 1e-8
            )

            actor_losses = []

            critic_losses = []

            for (
                log_prob,
                value,
                ret
            ) in zip(
                action_log_probs,
                critic_values,
                returns
            ):

                advantage = (
                    ret - value
                )

                actor_losses.append(
                    -log_prob
                    * tf.stop_gradient(
                        advantage
                    )
                )

                critic_losses.append(
                    tf.square(
                        advantage
                    )
                )

                ###############################################

                actor_loss_value = tf.reduce_mean(
                    actor_losses
                )

                critic_loss_value = tf.reduce_mean(
                    critic_losses
                )

                actor_loss_history.append(
                    float(actor_loss_value.numpy())
                )

                critic_loss_history.append(
                    float(critic_loss_value.numpy())
                )

            total_loss = (
                tf.add_n(actor_losses)
                +
                tf.add_n(critic_losses)
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

        # episode_reward = np.sum(
        #     rewards
        # )

        episode_reward = reward  # instead of np.sum(rewards)

        reward_history.append(
            float(episode_reward)
        )

        architecture_history.append(
            env.tree_encoding.copy()
        )

        if episode_reward > best_reward:

            best_reward = episode_reward

            best_architecture = (
                env.tree_encoding.copy()
            )

        print(
            f"Episode {episode+1} "
            f"Reward={episode_reward:.4f} "
            f"Arch={env.tree_encoding}"
        )

    return (
        model,
        best_architecture,
        best_reward,
        reward_history,
        actor_loss_history,
        critic_loss_history,
        architecture_history,
    )


def save_reward_plot(
    rewards,
    output_dir,
):
    plt.figure(figsize=(10,6))
    plt.plot(rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("NAS RL Reward")

    plt.grid(True)

    plt.savefig(
        os.path.join(
            output_dir,
            "reward_curve.png"
        )
    )

    plt.close()


def save_loss_plot(
    actor_losses,
    critic_losses,
    output_dir,
):

    plt.figure(figsize=(10,6))

    plt.plot(
        actor_losses,
        label="Actor"
    )

    plt.plot(
        critic_losses,
        label="Critic"
    )

    plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.title(
        "Actor Critic Loss"
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
    arr = np.full(
        (
            len(architectures),
            MAX_NODES
        ),
        -1,
        dtype=np.int32,
    )

    for i, arch in enumerate(
        architectures
    ):
        arr[
            i,
            :len(arch)
        ] = arch

    plt.figure(
        figsize=(12,8)
    )

    plt.imshow(
        arr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xlabel(
        "DFS Position"
    )

    plt.ylabel(
        "Episode"
    )

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
    reward,
    output_dir,
):

    plt.figure(
        figsize=(10,2)
    )

    plt.bar(
        np.arange(
            len(best_arch)
        ),
        best_arch
    )

    plt.xlabel(
        "DFS Node"
    )

    plt.ylabel(
        "Operation"
    )

    plt.title(
        f"Best Architecture Reward={reward:.4f}"
    )

    plt.savefig(
        os.path.join(
            output_dir,
            "best_architecture.png"
        )
    )

    plt.close()
################################################################################


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


def get_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.25):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    encoder_outputs = x + res

    return encoder_outputs #keras.Model(inputs, encoder_outputs, name="encoder")


def get_decoder(input_shape, mlp_units=[128], mlp_dropout=0.4):
    #inputs = keras.Input(shape=build_encoder_model(input_shape).output.shape[1:])
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(maxlen*num_actions, activation="relu")(x)
    # x = tf.reshape(x, (-1, maxlen, num_actions))
    # outputs = tf.argmax(x, axis=2)
    return keras.Model(inputs, outputs)


def build_encoder_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = keras.backend.expand_dims(inputs, axis=-1)
    for _ in range(num_transformer_blocks):
        x = get_encoder(x, head_size, num_heads, ff_dim, dropout)

    #output = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    output = x

    return keras.Model(inputs, output)


#######transformer position############################

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        super(TransformerBlock, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        if batch_size is None:
            batch_size = 1
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement an embedding layer
Create two seperate embedding layers: one for tokens and one for token index
(positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        #return positions



def get_vqvae(output_dim, latent_dim=NUM_HIDDEN, num_embeddings=maxlen, input_shape=maxlen):
    vq_layer = VectorQuantizer(latent_dim, num_embeddings, name="vector_quantizer")
    encoder = build_encoder_model(input_shape)
    decoder = get_decoder(input_shape)
    inputs = keras.Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    common = decoder(quantized_latents)

    action = layers.Dense(output_dim, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs, outputs=[action, critic], name="vq_vae")

    return model

################################################################################

def encoding_to_nats_arch(tree_encoding):
    """
    Convert your RL encoding into NATS adjacency matrix format.
    NATS expects:
      - 4 nodes (plus input/output implicit)
      - 6 edges upper-triangular
    """

    ops = ["none", "skip_connect", "conv_1x1", "conv_3x3"]

    arch = []

    for op in tree_encoding:
        if op == 0:
            arch.append(1)  # skip_connect
        elif op == 1:
            arch.append(2)  # conv_1x1
        else:
            arch.append(3)  # conv_3x3

    # pad/truncate to 6 edges (NATS-TSS requirement)
    arch = arch[:6]
    arch += [1] * (6 - len(arch))

    return tuple(arch)

################################################################################
def main():

    os.makedirs("results_a3c", exist_ok=True)

    tf.random.set_seed(42)
    np.random.seed(42)

    # -----------------------------
    # NATS-Bench API
    # -----------------------------
    api = create(
        "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
        "tss",
        fast_mode=True,
        verbose=False
    )

    env = NATSNASEnv(api, max_nodes=MAX_NODES)

    # -----------------------------
    # Train RL NAS controller
    # -----------------------------
    (
        model,
        best_arch,
        best_reward,
        reward_history,
        actor_loss_history,
        critic_loss_history,
        architecture_history,
    ) = train_nas_agent(
        env,
        episodes=100,
    )

    # -----------------------------
    # Save RL training artifacts (PLOTS)
    # -----------------------------
    save_reward_plot(reward_history, "results_a3c")
    save_loss_plot(actor_loss_history, critic_loss_history, "results_a3c")
    save_architecture_heatmap(architecture_history, "results_a3c")
    save_best_architecture(best_arch, best_reward, "results_a3c")

    # -----------------------------
    # Save run in JSON format (for your evaluation loader)
    # -----------------------------
    run_data = {
        "best_arch": best_arch,
        "best_reward": float(best_reward),
        "reward_history": [float(x) for x in reward_history],
        "actor_loss_history": [float(x) for x in actor_loss_history],
        "critic_loss_history": [float(x) for x in critic_loss_history],
    }

    run_path = os.path.join(
        "results_a3c",
        f"nats_a3c_run_{int(np.random.randint(0, 1e9))}.json"
    )

    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"Saved run to {run_path}")

    # -----------------------------
    # OPTIONAL: build final architecture (for inspection only)
    # -----------------------------
    print("\nBest architecture found:", best_arch)
    print("Best reward (NATS accuracy):", best_reward)

    print("\nTraining artifacts saved to ./results_a3c")




if __name__ == "__main__":
    main()