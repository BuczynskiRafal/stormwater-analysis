import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model  # noqa

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier = os.path.join(current_directory, "catchemnt_classifier", "model.keras")
recomendations_classifier = os.path.join(current_directory, "recomendations", "model.keras")


try:
    classifier = load_model(catchment_classifier)
except FileNotFoundError:
    print(f"Cannot load model: {catchment_classifier}")
    raise FileNotFoundError("Cannot load model")

try:
    recomendation = load_model(recomendations_classifier)
except FileNotFoundError:
    print(f"Cannot load model: {recomendations_classifier}")
    raise FileNotFoundError("Cannot load model")
