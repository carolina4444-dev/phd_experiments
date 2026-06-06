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
    output_layer = Dense(1, activation='sigmoid')(x)

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

# # Example usage
# input_shape = (250, 32)
# num_classes = 2  # Adjust based on your task
# tree_encoding = [0, 0, 0, 1, 1, 1]  # Adjust based on your tree structure

# model = build_tree_model(input_shape, num_classes, tree_encoding)
# model.summary()

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='tree_based_model.png', show_shapes=True, show_layer_names=True)