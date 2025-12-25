import os
import logging

# Disable GPU for consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model  # noqa

logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier_path = os.path.join(current_directory, "catchment_classifier", "model.keras")
recommendations_classifier_path = os.path.join(current_directory, "recommendations", "recomendations.keras")
gnn_model_path = os.path.join(current_directory, "recommendations", "graphsage_4hop_model.weights.h5")

try:
    classifier = load_model(catchment_classifier_path)
    logger.info(f"Catchment classifier loaded from: {catchment_classifier_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot load catchment classifier: {catchment_classifier_path}")

try:
    recommendation = load_model(recommendations_classifier_path)
    logger.info(f"MLP recommendation model loaded from: {recommendations_classifier_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot load MLP recommendation model: {recommendations_classifier_path}")

try:
    from .graph_constructor import load_gnn_model_weights

    gnn_recommendation = load_gnn_model_weights(gnn_model_path)
    if gnn_recommendation:
        logger.info(f"GNN model loaded successfully from {gnn_model_path}")
    else:
        logger.warning("GNN model loading failed")
        gnn_recommendation = None
except Exception as e:
    logger.warning(f"GNN model loading failed: {e}")
    gnn_recommendation = None

__all__ = ["classifier", "recommendation", "gnn_recommendation"]
