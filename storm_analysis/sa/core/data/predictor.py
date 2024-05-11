import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model  # noqa

current_directory = os.path.dirname(os.path.abspath(__file__))
catchment_classifier = os.path.join(current_directory, "catchemnt_classifier", "model.keras")
recomandatio_classifier = os.path.join(current_directory, "recomendations", "model.keras")


try:
    classifier = load_model(catchment_classifier)
except FileNotFoundError:
    print(f"Cannot load model: {catchment_classifier}")
    raise Exception("Cannot load model")

try:
    recomendation = load_model(recomandatio_classifier)
except FileExistsError:
    print(f"Cannot load model: {recomandatio_classifier}")
    raise Exception("Cannot load model")
