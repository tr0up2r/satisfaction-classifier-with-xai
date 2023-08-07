import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
from spacy.lang.en import English
import math

path = '/home/mykim/source/predicting-satisfaction-using-graphs'
nlp = English()
nlp.add_pipe("sentencizer")

model = SentenceTransformer('all-mpnet-base-v2')


def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5 * x)))


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    x = np.linspace(-10, 10, p_size)
    y = np.vectorize(rev_sigmoid)
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def text_segmentation(sentences, order=1):
    if len(sentences) < 3:
        return [' '.join(sentences)]
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    activated_similarities = activate_similarities(similarities, p_size=3)
    minmimas = argrelextrema(activated_similarities, np.less, order=order)

    split_points = [each for each in minmimas[0]]
    if not split_points:
        return [' '.join(sentences)]

    result = []
    text = ''
    for num, each in enumerate(sentences):
        text += ' ' + each
        if num in split_points:
            result.append(text)
            text = ''
    result.append(text)
    return result