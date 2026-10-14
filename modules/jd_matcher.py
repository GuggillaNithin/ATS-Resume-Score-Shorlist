import os
import re
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Exact common English stopwords from original implementation
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
}

_MODEL_CACHE = None

def get_sentence_transformer_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Get or initialize the SentenceTransformer model.
    Cached across calls for fast batch inference.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer(model_name)
    return _MODEL_CACHE

def clean_text(text: str) -> str:
    """
    Exact text cleaning logic from original app:
    Lowercase, remove special characters, remove stopwords, normalize spaces.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = [w for w in words if w not in STOPWORDS]
    return " ".join(cleaned_words)

def calculate_jd_match_score(jd_text: str, resume_text: str, model=None) -> float:
    """
    Calculate semantic similarity score (0.0 to 100.0) between JD and Resume.
    Reuses existing SentenceTransformer and cosine similarity logic.
    """
    if not jd_text or not resume_text:
        return 0.0

    if model is None:
        model = get_sentence_transformer_model()

    jd_cleaned = clean_text(jd_text)
    resume_cleaned = clean_text(resume_text)

    if not jd_cleaned or not resume_cleaned:
        return 0.0

    jd_embedding = model.encode(jd_cleaned)
    resume_embedding = model.encode(resume_cleaned)

    similarity = cosine_similarity(
        jd_embedding.reshape(1, -1),
        resume_embedding.reshape(1, -1)
    )[0][0]

    # Normalized to 0.0 - 100.0 scale, bounded between 0 and 100
    score = float(similarity) * 100.0
    return max(0.0, min(100.0, round(score, 2)))

def batch_calculate_match_scores(jd_text: str, resume_texts: list[str], model=None) -> list[float]:
    """
    High-performance batch encoding for scoring ~50 resumes simultaneously.
    Retains identical cosine similarity values while significantly reducing computation time.
    """
    if not resume_texts:
        return []

    if model is None:
        model = get_sentence_transformer_model()

    jd_cleaned = clean_text(jd_text)
    if not jd_cleaned:
        return [0.0] * len(resume_texts)

    cleaned_resumes = [clean_text(t) for t in resume_texts]
    valid_indices = [i for i, t in enumerate(cleaned_resumes) if t]

    scores = [0.0] * len(resume_texts)
    if not valid_indices:
        return scores

    # Encode JD and resumes in batch
    jd_embedding = model.encode([jd_cleaned])
    batch_resumes = [cleaned_resumes[i] for i in valid_indices]
    resume_embeddings = model.encode(batch_resumes, batch_size=32, show_progress_bar=False)

    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]
    for idx, sim in zip(valid_indices, similarities):
        score = float(sim) * 100.0
        scores[idx] = max(0.0, min(100.0, round(score, 2)))

    return scores

def get_screening_outcome(score: float, threshold: float) -> str:
    """Classify candidate based on score and threshold."""
    if score >= threshold:
        return "Meets Threshold"
    return "Below Threshold"
