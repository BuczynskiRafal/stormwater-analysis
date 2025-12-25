"""
Graph construction utilities for SWMM conduit networks.
Builds graphs where conduits are nodes, not junctions.
"""

import os
import logging
from typing import Optional, Any

import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from .constants import GNN_HIDDEN_UNITS, GNN_DROPOUT_RATE, GNN_AGGREGATOR
from .data_manager import get_default_feature_columns
from .enums import RecommendationCategory

logger = logging.getLogger(__name__)


class GraphSAGELayer(tf.keras.layers.Layer):
    """Proper GraphSAGE layer with message passing and aggregation."""

    def __init__(self, units, aggregator="mean", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.aggregator = aggregator
        self.use_bias = use_bias
        self.is_gnn = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.W_self = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_self", trainable=True)
        self.W_neigh = self.add_weight(
            shape=(input_dim, self.units), initializer="glorot_uniform", name="W_neigh", trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias", trainable=True)

        super().build(input_shape)

    def call(self, inputs, adjacency_matrix=None, training=False):
        """Forward pass with proper message passing."""

        if adjacency_matrix is None:
            result = tf.matmul(inputs, self.W_self)
            if self.use_bias:
                result += self.bias
            return tf.nn.relu(result)

        # Adjacency matrix must be preprocessed (normalized) before passing here
        if hasattr(adjacency_matrix, "toarray"):
            adj_dense = tf.constant(adjacency_matrix.toarray(), dtype=tf.float32)
        elif hasattr(adjacency_matrix, "indices"):
            adj_dense = tf.sparse.to_dense(adjacency_matrix, default_value=0.0)
            adj_dense = tf.cast(adj_dense, tf.float32)
        else:
            adj_dense = tf.cast(adjacency_matrix, tf.float32)

        neigh_feats = tf.matmul(adj_dense, inputs)
        self_transformed = tf.matmul(inputs, self.W_self)
        neigh_transformed = tf.matmul(neigh_feats, self.W_neigh)
        output = self_transformed + neigh_transformed
        if self.use_bias:
            output += self.bias

        return tf.nn.relu(output)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "aggregator": self.aggregator, "use_bias": self.use_bias})
        return config


class GraphSAGEModel(tf.keras.Model):
    """Pure GraphSAGE model with optimal hyperparameters from tuning."""

    def __init__(
        self,
        n_features,
        n_classes=9,
        aggregator=GNN_AGGREGATOR,
        hidden_units=GNN_HIDDEN_UNITS,
        dropout_rate=GNN_DROPOUT_RATE,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.graphsage1 = GraphSAGELayer(hidden_units, aggregator=aggregator)
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)

        self.graphsage2 = GraphSAGELayer(hidden_units, aggregator=aggregator)
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)

        self.output_layer = Dense(n_classes, activation="softmax")

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            node_features_shape = input_shape[0]
        else:
            node_features_shape = input_shape

        self.graphsage1.build(node_features_shape)
        self.graphsage2.build((node_features_shape[0], self.hidden_units))
        self.output_layer.build((node_features_shape[0], self.hidden_units))
        super().build(input_shape)

    def _supports_adjacency(self, layer):
        try:
            import inspect

            sig = inspect.signature(layer.call)
            return "adjacency_matrix" in sig.parameters
        except Exception:
            return False

    def _call_layer_safely(self, layer, inputs, adjacency, training):
        if self._supports_adjacency(layer):
            return layer(inputs, adjacency_matrix=adjacency, training=training)
        else:
            return layer(inputs, training=training)

    def call(self, inputs, training=False):
        if isinstance(inputs, list) and len(inputs) == 2:
            node_features, adjacency = inputs
        else:
            node_features = inputs
            adjacency = None

        x = self._call_layer_safely(self.graphsage1, node_features, adjacency, training)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self._call_layer_safely(self.graphsage2, x, adjacency, training)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        return self.output_layer(x)


class GNNModelLoadError(Exception):
    """Base exception for errors raised during GNN model loading."""

    pass


CUSTOM_OBJECTS = {"GraphSAGELayer": GraphSAGELayer, "GraphSAGEModel": GraphSAGEModel}


def load_gnn_model_weights(weights_path: str = None) -> Optional[Any]:
    """Load GNN model from .keras or reconstruct from .weights.h5."""
    project_root = os.path.abspath(os.path.dirname(__file__))

    keras_model_path = os.path.join(project_root, "recommendations", "graphsage_model.keras")
    if os.path.exists(keras_model_path):
        try:
            logger.info(f"Loading GNN model from: {keras_model_path}")
            model = tf.keras.models.load_model(keras_model_path, custom_objects=CUSTOM_OBJECTS)
            logger.info(f"Loaded GNN model from: {keras_model_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model from {keras_model_path}: {e}")

    model_weights_path = weights_path or os.path.join(project_root, "recommendations", "graphsage_model.weights.h5")
    try:
        logger.info(f"Loading GNN weights from: {model_weights_path}")
        n_features = len(get_default_feature_columns())
        n_classes = len(RecommendationCategory)

        model = GraphSAGEModel(n_features=n_features, n_classes=n_classes)
        model([tf.zeros((1, n_features)), tf.zeros((1, 1))], training=False)
        model.load_weights(model_weights_path)

        logger.info(f"Loaded GNN weights from: {model_weights_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_weights_path}")
    except (ValueError, OSError) as e:
        logger.error(
            f"Failed to load GNN model from {model_weights_path}. "
            f"The file may be corrupted or incompatible with model architecture. Error: {e}",
            exc_info=True,
        )
    except GNNModelLoadError as e:
        logger.error(
            f"An unexpected critical error occurred while loading the GNN model: {e}",
            exc_info=True,
        )
    return None
