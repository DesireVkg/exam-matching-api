import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.model_loader import model, tokenizer, label_encoder, curriculum
from app.utils import preprocess_text
import torch.nn.functional as F
import spacy
import re
from transformers import CamembertModel

nlp = spacy.load("fr_core_news_md")
embedding_model = CamembertModel.from_pretrained("camembert-base")

def get_embeddings(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embeddings.extend(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.array(embeddings)

def predict_exam_level(text, max_len=128):
    cleaned_text = preprocess_text(text)
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return predicted_class, probs

def analyze_exam(text):
    predicted_level, probs = predict_exam_level(text)
    topics = curriculum[predicted_level]
    
    text_embed = get_embeddings([preprocess_text(text)])[0]
    topic_embeds = get_embeddings(topics)
    
    sims = cosine_similarity([text_embed], topic_embeds)[0]
    covered = [(topic, round(float(sim), 3)) for topic, sim in zip(topics, sims) if sim >= 0.45]
    missing = [topic for topic, sim in zip(topics, sims) if sim < 0.45]
    avg_sim = np.mean([sim for sim in sims if sim >= 0.45]) if covered else 0.0

    sents = [s.text for s in nlp(preprocess_text(text)).sents if len(s.text) > 20]
    competencies = []
    if sents:
        sent_embeds = get_embeddings(sents)
        for s, e in zip(sents, sent_embeds):
            sim_scores = cosine_similarity([e], topic_embeds)[0]
            best = np.argmax(sim_scores)
            if sim_scores[best] >= 0.35:
                competencies.append(topics[best])
        competencies = sorted(set(competencies), key=lambda x: len(x), reverse=True)[:5]

    questions = [q.strip() for q in re.split(r'\d+[.)]|[-•]', text) if len(q.strip()) > 30]
    off_program = []
    if questions:
        q_embeds = get_embeddings(questions)
        for q, qe in zip(questions, q_embeds):
            max_sim = max(cosine_similarity([qe], topic_embeds)[0])
            if max_sim < 0.25:
                off_program.append((q[:150] + "...", round(float(max_sim), 3)))
        off_program = off_program[:5]

    def generate_recommendations(covered, missing, offp):
        recs = []
        if len(covered) < 3:
            recs.append("Renforcer la couverture des thèmes principaux du niveau")
        if missing:
            recs.append("Aborder les thèmes manquants: " + ", ".join(missing[:3]))
        if offp:
            recs.append("Réviser les questions hors programme détectées")
        return recs or ["Bon alignement avec le curriculum"]

    return {
        "predicted_level": predicted_level,
        "probabilities": dict(zip(label_encoder.classes_, map(float, probs))),
        "global_similarity": round(float(avg_sim), 3),
        "covered_themes": covered,
        "skills": competencies,
        "missing_themes": missing,
        "off_program": off_program,
        "recommendations": generate_recommendations([t[0] for t in covered], missing, [q[0] for q in off_program])
    }
