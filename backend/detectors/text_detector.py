"""
text_detector.py
-----------------
This file decides whether a piece of TEXT looks "AI-generated" or "human-written".

Two methods are used, in this order:

1. ONLINE method (optional): if the user has set up a free Hugging Face API
   token in the .env file, we send the text to a public AI-text-detection
   model and use its answer.

2. OFFLINE method (always available, no internet/API key needed): we run our
   own simple statistics on the text. This is the same basic idea that most
   free "AI detectors" use under the hood:

   - Burstiness: human writing has a mix of short and long sentences.
     AI writing tends to produce sentences that are more uniform in length.
   - Repetition: AI text often reuses the same words/phrases more than
     humans naturally do.
   - Vocabulary variety (Type-Token Ratio): how many UNIQUE words are used
     compared to the total number of words. Very high or very repetitive
     patterns can be a signal.
   - Average sentence length: AI text is often "smoother" and more
     consistent in structure.

IMPORTANT HONESTY NOTE:
These signals are estimates, not proof. No free tool can be 100% certain
whether text was written by AI. We are upfront about that in the UI.
"""

import os
import re
import statistics
import requests

# Free Hugging Face model that is specifically trained to spot AI text.
# (You can swap this for any other public text-classification model.)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"


def _split_sentences(text: str):
    """Break text into sentences using simple punctuation rules."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _split_words(text: str):
    """Break text into lowercase words, ignoring punctuation."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _offline_analysis(text: str):
    """
    Our own free, no-API statistics-based detector.
    Returns (label, confidence_percent, details_dict)
    """
    sentences = _split_sentences(text)
    words = _split_words(text)

    if len(words) < 5:
        return "Not enough text", 0, {
            "reason": "Please enter at least a few sentences for a meaningful result."
        }

    # 1. Sentence length list (in words)
    sentence_lengths = [len(_split_words(s)) for s in sentences if _split_words(s)]

    # 2. Burstiness = how much sentence lengths vary.
    #    Low variation (a low standard deviation) => more "AI-like" (uniform).
    if len(sentence_lengths) > 1:
        stdev = statistics.pstdev(sentence_lengths)
        mean_len = statistics.mean(sentence_lengths)
        # normalize: coefficient of variation (0 = totally uniform, higher = more human-like variety)
        burstiness = stdev / mean_len if mean_len > 0 else 0
    else:
        burstiness = 0.5  # not enough sentences to judge, assume neutral

    # 3. Vocabulary variety (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # 4. Repetition score: how often the most common words repeat
    from collections import Counter
    common = Counter(words).most_common(5)
    repetition_ratio = sum(c for _, c in common) / len(words)

    # ---- Combine signals into one AI-probability score (0 to 1) ----
    # Each signal nudges the score up (more AI-like) or down (more human-like).
    ai_score = 0.5  # start neutral

    # Low burstiness (uniform sentence lengths) -> more AI-like
    if burstiness < 0.3:
        ai_score += 0.20
    elif burstiness > 0.6:
        ai_score -= 0.15

    # Very "smooth"/average vocabulary variety -> more AI-like
    if 0.40 <= ttr <= 0.55:
        ai_score += 0.15
    elif ttr > 0.70:
        ai_score -= 0.15  # very rich, unusual vocabulary => human-like

    # High repetition of the same few words -> more AI-like (AI often over-uses
    # transition words / safe phrasing)
    if repetition_ratio > 0.18:
        ai_score += 0.10
    else:
        ai_score -= 0.05

    # Clamp between 0.05 and 0.95 (we never claim 100% certainty)
    ai_score = max(0.05, min(0.95, ai_score))

    label = "Likely AI Generated" if ai_score >= 0.5 else "Likely Human Created"
    confidence = round(ai_score * 100) if ai_score >= 0.5 else round((1 - ai_score) * 100)

    details = {
        "method": "offline-statistical",
        "sentence_count": len(sentences),
        "word_count": len(words),
        "average_sentence_length": round(statistics.mean(sentence_lengths), 1) if sentence_lengths else 0,
        "vocabulary_variety_percent": round(ttr * 100, 1),
        "repetition_percent": round(repetition_ratio * 100, 1),
    }

    return label, confidence, details


def _online_analysis(text: str, hf_token: str):
    """
    Calls the free Hugging Face Inference API for a model trained to
    detect AI-generated text. Returns None if it fails for any reason
    (no internet, no token, model asleep, etc.) so we can fall back
    to the offline method.
    """
    try:
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.post(
            HF_MODEL_URL,
            headers=headers,
            json={"inputs": text[:2000]},  # keep requests small/fast
            timeout=15,
        )
        if response.status_code != 200:
            return None

        data = response.json()
        # Expected shape: [[{"label": "Fake", "score": 0.9}, {"label": "Real", "score": 0.1}]]
        scores = data[0] if isinstance(data, list) else None
        if not scores:
            return None

        fake_score = next((s["score"] for s in scores if s["label"].lower() in ("fake", "ai")), None)
        if fake_score is None:
            return None

        label = "Likely AI Generated" if fake_score >= 0.5 else "Likely Human Created"
        confidence = round(fake_score * 100) if fake_score >= 0.5 else round((1 - fake_score) * 100)
        return label, confidence, {"method": "huggingface-roberta-openai-detector"}

    except Exception:
        return None


def detect_text(text: str):
    """
    Main function called by app.py.
    Tries the online model first (if a token is set), otherwise uses the
    offline statistical method.
    """
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()

    if hf_token:
        result = _online_analysis(text, hf_token)
        if result:
            label, confidence, details = result
            return {"result": label, "confidence": confidence, "details": details}

    # Fallback (or default if no token was provided)
    label, confidence, details = _offline_analysis(text)
    return {"result": label, "confidence": confidence, "details": details}
