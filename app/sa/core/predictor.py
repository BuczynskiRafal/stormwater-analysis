import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier_path = os.path.join(current_directory, "catchment_classifier", "model.keras")
recommendations_classifier_path = os.path.join(current_directory, "recommendations", "recomendations.keras")
gnn_model_path = os.path.join(current_directory, "recommendations", "graphsage_4hop_model.weights.h5")

# Lazy-loaded models
_classifier: Optional[Any] = None
_recommendation: Optional[Any] = None
_gnn_recommendation: Optional[Any] = None
_models_loaded = False


def _load_models() -> None:
    """Lazy load TensorFlow models on first access."""
    global _classifier, _recommendation, _gnn_recommendation, _models_loaded

    if _models_loaded:
        return

    _models_loaded = True

    try:
        # Disable GPU for consistency
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        from tensorflow.keras.models import load_model

        try:
            _classifier = load_model(catchment_classifier_path)
            logger.info(f"Catchment classifier loaded from: {catchment_classifier_path}")
        except FileNotFoundError:
            logger.error(f"Cannot load catchment classifier: {catchment_classifier_path}")

        try:
            _recommendation = load_model(recommendations_classifier_path)
            logger.info(f"MLP recommendation model loaded from: {recommendations_classifier_path}")
        except FileNotFoundError:
            logger.error(f"Cannot load MLP recommendation model: {recommendations_classifier_path}")

        try:
            from .graph_constructor import load_gnn_model_weights

            _gnn_recommendation = load_gnn_model_weights(gnn_model_path)
            if _gnn_recommendation:
                logger.info(f"GNN model loaded successfully from {gnn_model_path}")
            else:
                logger.warning("GNN model loading failed")
        except Exception as e:
            logger.warning(f"GNN model loading failed: {e}")

    except ImportError:
        logger.warning("TensorFlow not available - models will not be loaded")


def get_classifier() -> Optional[Any]:
    """Get the catchment classifier model."""
    _load_models()
    return _classifier


def get_recommendation() -> Optional[Any]:
    """Get the MLP recommendation model."""
    _load_models()
    return _recommendation


def get_gnn_recommendation() -> Optional[Any]:
    """Get the GNN recommendation model."""
    _load_models()
    return _gnn_recommendation


# Backward compatibility - lazy properties
class _LazyModelProxy:
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name):
        model = self._getter()
        if model is None:
            raise RuntimeError("Model not loaded - TensorFlow may not be available")
        return getattr(model, name)

    def __call__(self, *args, **kwargs):
        model = self._getter()
        if model is None:
            raise RuntimeError("Model not loaded - TensorFlow may not be available")
        return model(*args, **kwargs)


classifier = _LazyModelProxy(get_classifier)
recommendation = _LazyModelProxy(get_recommendation)
gnn_recommendation = _LazyModelProxy(get_gnn_recommendation)

__all__ = ["classifier", "recommendation", "gnn_recommendation", "get_classifier", "get_recommendation", "get_gnn_recommendation"]
