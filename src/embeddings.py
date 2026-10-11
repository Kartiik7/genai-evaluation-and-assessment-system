import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# In-memory cache for loaded models
_model_cache = {}

def get_model(name):
    """Loads a SentenceTransformer model by name and caches it in memory."""
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]

def embed_texts(model_name, texts):
    """
    Returns normalized embeddings as a numpy array.
    Caches embeddings to data/processed/emb_cache/ keyed by model name and a hash of the text.
    """
    safe_model_name = model_name.replace("/", "_")
    cache_dir = os.path.join("data", "processed", "emb_cache", safe_model_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    embeddings = []
    texts_to_compute = []
    indices_to_compute = []
    
    # Check on-disk cache for each text
    for i, text in enumerate(texts):
        # Ensure text is a string
        text_str = str(text) if text is not None else ""
        text_hash = hashlib.md5(text_str.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.npy")
        
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            embeddings.append(emb)
        else:
            embeddings.append(None) # Placeholder
            texts_to_compute.append(text_str)
            indices_to_compute.append((i, cache_path))
            
    # Compute and cache any missing embeddings
    if texts_to_compute:
        model = get_model(model_name)
        new_embs = model.encode(texts_to_compute, normalize_embeddings=True)
        
        for emb, (orig_idx, cache_path) in zip(new_embs, indices_to_compute):
            embeddings[orig_idx] = emb
            np.save(cache_path, emb)
            
    return np.array(embeddings)

def cosine_scores(model_name, references, answers):
    """
    Returns the cosine similarity between each reference[i] and answer[i], clipped to [0, 1].
    """
    ref_embs = embed_texts(model_name, references)
    ans_embs = embed_texts(model_name, answers)
    
    # Element-wise dot product across the embedding dimension (axis=1)
    # Since embeddings are L2 normalized, dot product equals cosine similarity
    scores = np.sum(ref_embs * ans_embs, axis=1)
    
    return np.clip(scores, 0.0, 1.0)
