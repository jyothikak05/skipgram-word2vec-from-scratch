import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PATHS ----------
EMB_PATH = "results/embeddings.pt"
WORD2IDX_PATH = "results/word2idx.pkl"
IDX2WORD_PATH = "results/idx2word.pkl"
OUTPUT_PATH = "results/analogy_results.txt"

# Load embeddings
embeddings = torch.load(EMB_PATH).numpy()

with open(WORD2IDX_PATH, "rb") as f:
    word2idx = pickle.load(f)

with open(IDX2WORD_PATH, "rb") as f:
    idx2word = pickle.load(f)

def analogy(a, b, c, top_k=5):
    """
    Solves analogy: b - a + c
    """
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []

    va = embeddings[word2idx[a]]
    vb = embeddings[word2idx[b]]
    vc = embeddings[word2idx[c]]

    target = vb - va + vc
    sims = cosine_similarity([target], embeddings)[0]

    # Remove input words from results
    for w in [a, b, c]:
        sims[word2idx[w]] = -1

    best = np.argsort(sims)[-top_k:][::-1]
    return [(idx2word[i], sims[i]) for i in best]

# --------- ANALOGY TESTS ---------
tests = [
    ("man", "king", "woman"),
    ("france", "paris", "italy"),
    ("big", "bigger", "small"),
    ("good", "better", "bad"),
]

lines = []

for a, b, c in tests:
    result = analogy(a, b, c, top_k=3)
    lines.append(f"{b} - {a} + {c}")
    for word, score in result:
        lines.append(f"  {word} ({score:.4f})")
    lines.append("")

    print(f"{b} - {a} + {c} → {result[0][0]}")

with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(lines))

print("\nAnalogy results saved.")
