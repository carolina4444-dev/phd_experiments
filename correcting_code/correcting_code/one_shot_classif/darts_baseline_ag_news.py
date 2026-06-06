"""
DARTS NAS Problem
=================

Adapted directly from the provided tree-based
Inception/Conv1D architecture generator.

Main idea
---------
Instead of:
    tree_encoding = [0,1,0,1...]

we relax architecture selection continuously
using DARTS mixed operations.

Search Space
-------------
Each searchable node chooses between:

0 -> Inception Module
1 -> Conv1D + MaxPool block

using differentiable architecture parameters.

This implementation cleanly separates:
    - NAS Problem
    - Search Space
    - Search Procedure

0 -> Inception
0 -> Inception
1 -> ConvLeaf
1 -> ConvLeaf
0 -> Inception
1 -> ConvLeaf
2 -> stop
=========================================================
"""

import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Concatenate,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# =========================================================
# TOKEN + POSITION EMBEDDING
# =========================================================

class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(
        self,
        maxlen,
        vocab_size,
        embed_dim,
    ):

        super().__init__()

        self.maxlen = maxlen

        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
        )

        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
        )

    def call(self, x):

        maxlen = tf.shape(x)[-1]

        positions = tf.range(
            start=0,
            limit=maxlen,
            delta=1
        )

        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions

# =========================================================
# ABSTRACT NAS PROBLEM
# =========================================================

class NASProblem:

    def search(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def export_architecture(self):
        raise NotImplementedError


# =========================================================
# INCEPTION BLOCK LAYER
# =========================================================

class InceptionBlock(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()

        self.branch1x1 = Conv1D(
            64,
            1,
            padding='same',
            activation='relu'
        )

        self.branch3x3_reduce = Conv1D(
            128,
            1,
            padding='same',
            activation='relu'
        )

        self.branch3x3 = Conv1D(
            128,
            3,
            padding='same',
            activation='relu'
        )

        self.branch5x5_reduce = Conv1D(
            32,
            1,
            padding='same',
            activation='relu'
        )

        self.branch5x5 = Conv1D(
            32,
            5,
            padding='same',
            activation='relu'
        )

        self.branch_pool = MaxPooling1D(
            3,
            strides=1,
            padding='same'
        )

        self.branch_pool_proj = Conv1D(
            32,
            1,
            padding='same',
            activation='relu'
        )

        self.concat = Concatenate(axis=-1)

    def call(self, x):

        b1 = self.branch1x1(x)

        b3 = self.branch3x3_reduce(x)
        b3 = self.branch3x3(b3)

        b5 = self.branch5x5_reduce(x)
        b5 = self.branch5x5(b5)

        bp = self.branch_pool(x)
        bp = self.branch_pool_proj(bp)

        return self.concat([b1, b3, b5, bp])


# =========================================================
# CONV LEAF BLOCK
# =========================================================

class ConvLeafBlock(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()

        self.conv = Conv1D(
            64,
            3,
            padding='same',
            activation='relu'
        )

        self.pool = MaxPooling1D(
            3,
            strides=1,
            padding='same'
        )

    def call(self, x):

        x = self.conv(x)

        return self.pool(x)


# =========================================================
# MIXED OPERATION
# =========================================================

class MixedOperation(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------
        # candidate operations
        # ---------------------------------------------

        self.inception = InceptionBlock()

        self.conv_leaf = ConvLeafBlock()

        # ---------------------------------------------
        # projection layers
        # ---------------------------------------------

        self.proj_inception = Conv1D(
            256,
            1,
            padding='same'
        )

        self.proj_conv = Conv1D(
            256,
            1,
            padding='same'
        )

        # ---------------------------------------------
        # architecture parameters
        # ---------------------------------------------

        self.alpha = self.add_weight(
            name="alpha",
            shape=(2,),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=1e-3
            ),
            trainable=True,
        )

    def call(self, x):

        weights = tf.nn.softmax(self.alpha)

        op0 = self.inception(x)

        op1 = self.conv_leaf(x)

        op0 = self.proj_inception(op0)

        op1 = self.proj_conv(op1)

        return (
            weights[0] * op0
            + weights[1] * op1
        )

# =========================================================
# DARTS SEARCH NETWORK
# =========================================================

class DARTSSearchModel(Model):

    def __init__(
        self,
        input_shape,
        max_length,
        vocab_size,
        embedding_dim,
        num_search_nodes=6,
    ):

        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            max_length,
            vocab_size,
            embedding_dim
        )

        self.search_nodes = [
            MixedOperation()
            for _ in range(num_search_nodes)
        ]

        self.global_pool = GlobalAveragePooling1D()

        self.fc = Dense(
            256,
            activation='relu'
        )

        self.dropout = Dropout(0.5)

        self.output_layer = Dense(
            4,
            activation='softmax'
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def call(self, x, training=False):

        x = self.embedding(x)

        for node in self.search_nodes:

            x = node(x)

        x = self.global_pool(x)

        x = self.fc(x)

        x = self.dropout(
            x,
            training=training
        )

        return self.output_layer(x)

    # =====================================================
    # ARCHITECTURE PARAMETERS
    # =====================================================

    @property
    def arch_parameters(self):

        return [
            node.alpha
            for node in self.search_nodes
        ]

    # =====================================================
    # WEIGHT PARAMETERS
    # =====================================================

    @property
    def weight_parameters(self):

        arch_ids = {
            id(p)
            for p in self.arch_parameters
        }

        return [
            p
            for p in self.trainable_variables
            if id(p) not in arch_ids
        ]


# =========================================================
# DARTS NAS PROBLEM
# =========================================================

class DARTSNASProblem(NASProblem):

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        input_shape,
        max_length,
        vocab_size,
        embedding_dim,
    ):
        self.max_tree_length = 5

        self.train_dataset = train_dataset

        self.valid_dataset = valid_dataset

        self.model = DARTSSearchModel(
            input_shape=input_shape,
            max_length=max_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        # =============================================
        # OPTIMIZERS
        # =============================================

        self.weight_optimizer = tf.keras.optimizers.Adam(
            1e-3
        )

        self.arch_optimizer = tf.keras.optimizers.Adam(
            3e-4
        )

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        epochs=50,
        steps_per_epoch=20
    ):

        train_iter = iter(
            self.train_dataset
        )

        valid_iter = iter(
            self.valid_dataset.repeat()
        )

        history = {
            "train_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):

            self.metric.reset_state()

            epoch_losses = []

            for _ in range(steps_per_epoch):

                train_batch = next(train_iter)

                valid_batch = next(valid_iter)

                loss = self._train_step(
                    train_batch,
                    valid_batch
                )

                epoch_losses.append(
                    float(loss.numpy())
                )

            mean_loss = sum(epoch_losses) / len(epoch_losses)

            acc = float(
                self.metric.result().numpy()
            )

            history["train_loss"].append(
                mean_loss
            )

            history["val_acc"].append(
                acc
            )

            print(
                f"Epoch={epoch+1} "
                f"Loss={mean_loss:.4f} "
                f"ValAcc={acc:.4f}"
            )

        return history

    # =====================================================
    # DARTS STEP
    # =====================================================

    @tf.function
    def _train_step(
        self,
        train_batch,
        valid_batch,
    ):

        train_x, train_y = train_batch
        valid_x, valid_y = valid_batch

        # =============================================
        # UPDATE NETWORK WEIGHTS
        # =============================================

        with tf.GradientTape() as tape:

            logits = self.model(
                train_x,
                training=True
            )

            loss = self.loss_fn(
                train_y,
                logits
            )

        grads = tape.gradient(
            loss,
            self.model.weight_parameters
        )

        self.weight_optimizer.apply_gradients(
            zip(
                grads,
                self.model.weight_parameters
            )
        )

        # =============================================
        # UPDATE ARCHITECTURE PARAMETERS
        # =============================================

        with tf.GradientTape() as tape:

            logits = self.model(
                valid_x,
                training=True
            )

            arch_loss = self.loss_fn(
                valid_y,
                logits
            )

        arch_grads = tape.gradient(
            arch_loss,
            self.model.arch_parameters
        )

        self.arch_optimizer.apply_gradients(
            zip(
                arch_grads,
                self.model.arch_parameters
            )
        )

        self.metric.update_state(
            valid_y,
            logits
        )

        return loss

    # =====================================================
    # EXPORT DISCRETE ARCHITECTURE
    # =====================================================

    def export_architecture(self):

        architecture = []

        for alpha in self.model.arch_parameters:

            probs = tf.nn.softmax(alpha)

            op = int(tf.argmax(probs))

            architecture.append(op)

        # ----------------------------------
        # DFS padding
        # ----------------------------------

        while len(architecture) < self.max_tree_length:
            architecture.append(2)

        return architecture[:self.max_tree_length]

    # =====================================================
    # BUILD FINAL DISCRETE MODEL
    # =====================================================

    def inception_module(self, x):

        branch1x1 = Conv1D(
            64,
            1,
            padding='same',
            activation='relu'
        )(x)

        branch3x3 = Conv1D(
            128,
            1,
            padding='same',
            activation='relu'
        )(x)

        branch3x3 = Conv1D(
            128,
            3,
            padding='same',
            activation='relu'
        )(branch3x3)

        branch5x5 = Conv1D(
            32,
            1,
            padding='same',
            activation='relu'
        )(x)

        branch5x5 = Conv1D(
            32,
            5,
            padding='same',
            activation='relu'
        )(branch5x5)

        branch_pool = MaxPooling1D(
            3,
            strides=1,
            padding='same'
        )(x)

        branch_pool = Conv1D(
            32,
            1,
            padding='same',
            activation='relu'
        )(branch_pool)

        return Concatenate(axis=-1)([
            branch1x1,
            branch3x3,
            branch5x5,
            branch_pool
        ])


    def conv_leaf_module(self, x):

        x = Conv1D(
            64,
            3,
            padding='same',
            activation='relu'
        )(x)

        x = MaxPooling1D(
            3,
            strides=1,
            padding='same'
        )(x)

        return x


    def build_final_model(self):

        encoding = self.export_architecture()

        print("Selected encoding:")
        print(encoding)

        input_layer = Input(shape=(self.model.embedding.maxlen,))

        x = input_layer

        x = self.model.embedding(x)

        for bit in encoding:

            if bit == 2:
                break

            elif bit == 0:
                x = self.inception_module(x)

            elif bit == 1:
                x = self.conv_leaf_module(x)

        x = GlobalAveragePooling1D()(x)

        x = Dense(
            256,
            activation='relu'
        )(x)

        output = Dense(
            4,
            activation='softmax'
        )(x)

        return Model(
            inputs=input_layer,
            outputs=output
        )

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(self):

        architecture = self.export_architecture()

        print("\nLearned Architecture:")
        print(architecture)

        return architecture


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":

    VOCAB_SIZE = 20000
    MAX_LENGTH = 250

    NUM_CLASSES = 4

    SHOTS_PER_CLASS = 5
    VALID_SIZE = 5000

    BATCH_SIZE = 16

    SEARCH_EPOCHS = 50
    STEPS_PER_EPOCH = 20

    from datasets import load_dataset

    # =============================================
    # LOAD AG NEWS
    # =============================================

    dataset = load_dataset("ag_news")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]


    VOCAB_SIZE = 20000
    MAX_LENGTH = 250

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH,
    )

    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(train_texts)
        .batch(256)
    )

    # 0 = World
    # 1 = Sports
    # 2 = Business
    # 3 = Sci/Tech

    # -------------------------------------------------
    # CONFIG
    # -------------------------------------------------

    VOCAB_SIZE = 20000
    MAX_LENGTH = 250

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH,
    )

    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(train_texts)
        .batch(256)
    )

    x_train = vectorizer(
        tf.constant(train_texts)
    )

    x_test = vectorizer(
        tf.constant(test_texts)
    )

    y_train = tf.constant(train_labels)
    y_test = tf.constant(test_labels)

    NUM_CLASSES = 4

    support_indices = []

    for cls in range(NUM_CLASSES):

        class_indices = tf.where(
            tf.equal(y_train, cls)
        )[:, 0]

        selected = class_indices[:SHOTS_PER_CLASS]

        support_indices.extend(
            selected.numpy().tolist()
        )

    support_indices = tf.constant(
        support_indices,
        dtype=tf.int32
    )

    x_train_final = tf.gather(
        x_train,
        support_indices
    )

    y_train_final = tf.gather(
        y_train,
        support_indices
    )

    print(
        f"Few-shot training samples: "
        f"{len(x_train_final)}"
    )

    # -------------------------------------------------
    # TRAIN / VALID SPLIT
    # -------------------------------------------------
    mask = tf.ones(
        len(x_train),
        dtype=tf.bool
    )

    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.expand_dims(
            support_indices,
            axis=1
        ),
        tf.zeros(
            len(support_indices),
            dtype=tf.bool
        )
    )

    remaining_x = tf.boolean_mask(
        x_train,
        mask
    )

    remaining_y = tf.boolean_mask(
        y_train,
        mask
    )

    x_valid = remaining_x[:VALID_SIZE]
    y_valid = remaining_y[:VALID_SIZE]

    # -------------------------------------------------
    # TF DATASETS
    # -------------------------------------------------

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_train_final,
                y_train_final
            )
        )
        .shuffle(
            len(x_train_final)
        )
        .repeat()
        .batch(
            min(
                BATCH_SIZE,
                len(x_train_final)
            )
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_valid,
                y_valid
            )
        )
        .batch(64)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_test,
                y_test
            )
        )
        .batch(64)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    # =============================================
    # CREATE NAS PROBLEM
    # =============================================

    problem = DARTSNASProblem(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        input_shape=(MAX_LENGTH,),
        max_length=MAX_LENGTH,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
    )

    # =============================================
    # START SEARCH
    # =============================================

    search_history = problem.search(
        epochs=SEARCH_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH
    )

    plt.figure(figsize=(8,5))

    plt.plot(
        search_history["train_loss"],
        label="Search Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DARTS Search Loss")
    plt.legend()

    plt.savefig(
        "darts_search_loss.png",
        bbox_inches="tight"
    )

    plt.close()

    # =============================================
    # EXPORT BEST ARCHITECTURE
    # =============================================

    architecture = problem.evaluate()

    print("\nSelected architecture:")
    print(architecture)

    # =============================================
    # BUILD FINAL MODEL
    # =============================================

    final_model = problem.build_final_model()

    final_model.summary()

    final_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    final_history = final_model.fit(
        train_dataset,
        epochs=20,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset
    )

    test_loss, test_acc = final_model.evaluate(
        test_dataset,
        verbose=1
    )

    print("\nFINAL TEST RESULTS")
    print(
        f"Test Loss: {test_loss:.4f}"
    )
    print(
        f"Test Accuracy: {test_acc:.4f}"
    )

    plt.figure(figsize=(8,5))

    plt.plot(
        final_history.history["loss"],
        label="Train Loss"
    )

    if "val_loss" in final_history.history:

        plt.plot(
            final_history.history["val_loss"],
            label="Validation Loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Final Model Loss"
    )

    plt.legend()

    plt.savefig(
        "final_model_loss.png",
        bbox_inches="tight"
    )

    plt.close()

    import pandas as pd

    pd.DataFrame(
        search_history
    ).to_csv(
        "search_metrics.csv",
        index=False
    )

    pd.DataFrame(
        final_history.history
    ).to_csv(
        "final_model_metrics.csv",
        index=False
    )

    pd.DataFrame(
        {
            "test_loss": [test_loss],
            "test_accuracy": [test_acc]
        }
    ).to_csv(
        "test_results.csv",
        index=False
    )

"""
final files:

search_metrics.csv

final_model_metrics.csv

test_results.csv

darts_search_loss.png

darts_search_accuracy.png

final_model_loss.png

final_model_accuracy.png

"""
