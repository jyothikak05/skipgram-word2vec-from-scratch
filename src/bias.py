import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PATHS ----------
EMB_PATH = "results/embeddings.pt"
WORD2IDX_PATH = "results/word2idx.pkl"
IDX2WORD_PATH = "results/idx2word.pkl"
OUTPUT_PATH = "results/bias_scores.txt"

# Load embeddings
embeddings = torch.load(EMB_PATH).numpy()

with open(WORD2IDX_PATH, "rb") as f:
    word2idx = pickle.load(f)

with open(IDX2WORD_PATH, "rb") as f:
    idx2word = pickle.load(f)

def get_vector(word):
    return embeddings[word2idx[word]]

def cosine(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# --------- STEP 1: Gender Direction ---------
gender_pairs = [
    ("man", "woman"),
    ("he", "she"),
    ("boy", "girl"),
]

gender_vectors = []

for w1, w2 in gender_pairs:
    if w1 in word2idx and w2 in word2idx:
        gender_vectors.append(get_vector(w1) - get_vector(w2))

gender_direction = np.mean(gender_vectors, axis=0)
gender_direction /= np.linalg.norm(gender_direction)

# --------- STEP 2: Bias Evaluation ---------
profession_words = [
    "engineer", "doctor", "scientist", "programmer",
    "nurse", "teacher", "assistant", "receptionist",
    "lawyer", "manager"
]

lines = []
lines.append("Word\tBias Score")
lines.append("-------------------")

for word in profession_words:
    if word in word2idx:
        score = cosine(get_vector(word), gender_direction)
        lines.append(f"{word}\t{score:.4f}")
        print(f"{word}: {score:.4f}")

with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(lines))

print("\nBias scores saved.")
