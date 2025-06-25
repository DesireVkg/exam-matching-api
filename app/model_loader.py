import torch
import joblib
from transformers import CamembertTokenizer, CamembertModel
from app.utils import CamembertClassifier
from app.curriculum import curriculum  # Tu as ce dict dans ton script

# Chargement
tokenizer = CamembertTokenizer.from_pretrained("camembert_tokenizer")
label_encoder = joblib.load("label_encoder.pkl")

# Chargement modèle
model = CamembertClassifier(CamembertModel.from_pretrained("camembert-base"),
                            num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("camembert_classifier.pth", map_location="cpu"))
model.eval()
