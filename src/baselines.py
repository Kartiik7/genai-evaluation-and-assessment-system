import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

def calculate_tfidf_cosine(reference_answers, student_answers, fit_corpus=None):
    """
    Calculates TF-IDF cosine similarity between reference and student answers.
    
    Args:
        reference_answers: List/Series of reference answers
        student_answers: List/Series of student answers
        fit_corpus: List/Series of text to fit the TF-IDF vectorizer on. 
                    If None, fits on the combined reference and student answers provided.
                    
    Returns:
        numpy array of cosine similarities (0 to 1)
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    
    if fit_corpus is not None:
        vectorizer.fit(fit_corpus)
    else:
        # Combine both for fitting if no separate corpus is provided
        vectorizer.fit(list(reference_answers) + list(student_answers))
        
    ref_vecs = vectorizer.transform(reference_answers)
    student_vecs = vectorizer.transform(student_answers)
    
    # paired_cosine_distances returns distance (0 is identical, 1 is orthogonal).
    # Similarity = 1 - distance
    distances = paired_cosine_distances(ref_vecs, student_vecs)
    similarities = 1 - distances
    
    # Just in case of floating point inaccuracies
    return np.clip(similarities, 0, 1)
