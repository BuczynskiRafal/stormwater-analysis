import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model  # noqa

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier_path = os.path.join(current_directory, "catchment_classifier", "model.keras")
recommendations_classifier_path = os.path.join(current_directory, "recommendations", "recomendations.keras")

try:
    classifier = load_model(catchment_classifier_path)
except FileNotFoundError:
    print(f"Cannot load model: {catchment_classifier_path}")
    raise FileNotFoundError("Cannot load model")

try:
    recommendation = load_model(recommendations_classifier_path)
except FileNotFoundError:
    print(f"Cannot load model: {recommendations_classifier_path}")
    raise FileNotFoundError("Cannot load model")
