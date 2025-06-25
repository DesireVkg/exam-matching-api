import re
import spacy
import torch.nn as nn

nlp = spacy.load("fr_core_news_md")

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct])

class CamembertClassifier(nn.Module):
    def __init__(self, camembert_model, num_classes):
        super(CamembertClassifier, self).__init__()
        self.camembert = camembert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(self.dropout(pooled))
