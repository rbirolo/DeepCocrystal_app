# %%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#DeepCrystal Architecture

class DeepCrystal:
    def __init__(
        self,
        n_cnn: int,
        n_filters: int,
        kernel_sizes: int,
        cnn_activation: str,
        n_dense: int,
        dense_layer_size: int,
        dense_activation: str,
        embedding_dim: int,
        dropout_rate: float,
        optimizer_name: str,
        learning_rate: float,
        batch_size: int,
        n_epochs: int,
        max_api_len: int = 80,
        max_cof_len: int = 80,
        csv_log_path: str = None,
    ):
        self.n_cnn = n_cnn
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.cnn_activation = cnn_activation
        self.n_dense = n_dense
        self.dense_layer_size = dense_layer_size
        self.dense_activation = dense_activation
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_api_len = max_api_len
        self.max_cof_len = max_cof_len
        self.csv_log_path = csv_log_path
        self.vocab_size = 30

        csv_log_dir = self.csv_log_path[: self.csv_log_path.rfind("/")]
        os.makedirs(csv_log_dir, exist_ok=True)

    def build_model(self):
        api_input = keras.layers.Input(shape=(self.max_api_len,), dtype="int32")
        api_representation = keras.layers.Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_api_len,
            mask_zero=True,
        )(api_input)
        cof_input = keras.layers.Input(shape=(self.max_cof_len,), dtype="int32")
        cof_representation = keras.layers.Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_cof_len,
            mask_zero=True,
        )(cof_input)

        for cnn_idx in range(self.n_cnn):
            api_representation = keras.layers.Conv1D(
                filters=self.n_filters * (cnn_idx + 1),
                kernel_size=self.kernel_sizes,
                activation=self.cnn_activation,
                padding="valid",
                strides=1,
                name=f"api_cnn_{cnn_idx}",
            )(api_representation)
            cof_representation = keras.layers.Conv1D(
                filters=self.n_filters * (cnn_idx + 1),
                kernel_size=self.kernel_sizes,
                activation=self.cnn_activation,
                padding="valid",
                strides=1,
                name=f"cof_cnn_{cnn_idx}",
            )(cof_representation)

        api_representation = keras.layers.GlobalMaxPooling1D()(api_representation)
        cof_representation = keras.layers.GlobalMaxPooling1D()(cof_representation)

        interaction_representation = keras.layers.Concatenate(axis=-1)(
            [api_representation, cof_representation]
        )

        for dense_idx in range(self.n_dense):
            interaction_representation = keras.layers.Dense(
                self.dense_layer_size,
                activation=self.dense_activation,
                name=f"interaction_dense_{dense_idx}",
            )(interaction_representation)
            interaction_representation = keras.layers.Dropout(self.dropout_rate)(
                interaction_representation
            )

        predictions = keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer="normal"
        )(interaction_representation)

        if self.optimizer_name == "adam":
            opt = keras.optimizers.Adam(self.learning_rate)

        model = keras.models.Model(inputs=[api_input, cof_input], outputs=[predictions])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        train_api,
        train_coformer,
        train_labels,
    ):
        history = self.model.fit(
            x=[train_api, train_coformer],
            y=train_labels,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="accuracy",
                    min_delta=1e-5,
                    patience=5,
                    verbose=1,
                    restore_best_weights=False,
                ),
                tf.keras.callbacks.CSVLogger(self.csv_log_path),
            ],
        ).history
        return history

    def predict(
        self,
        test_api,
        test_coformer,
    ):
        predictions = self.model.predict(
            x=[test_api, test_coformer],
        )
        return predictions
    
# %%
