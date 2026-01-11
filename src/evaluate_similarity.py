import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import gensim.downloader as api

# ---------- PATHS ----------
EMB_PATH = "results/embeddings.pt"
WORD2IDX_PATH = "results/word2idx.pkl"
IDX2WORD_PATH = "results/idx2word.pkl"
OUTPUT_PATH = "results/cosine_similarity.txt"

# Load your embeddings
embeddings = torch.load(EMB_PATH).numpy()

with open(WORD2IDX_PATH, "rb") as f:
    word2idx = pickle.load(f)

with open(IDX2WORD_PATH, "rb") as f:
    idx2word = pickle.load(f)

# Load Gensim pretrained vectors
# Download once: GoogleNews-vectors-negative300.bin.gz
gensim_model = api.load("word2vec-google-news-300")

def cosine(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Word pairs to compare
pairs = [
    ("king", "queen"),
    ("paris", "france"),
    ("man", "woman"),
    ("computer", "laptop"),
    ("big", "small")
]

lines = []
lines.append("Word Pair | Our Model | Gensim")
lines.append("----------------------------------")

for w1, w2 in pairs:
    if w1 in word2idx and w2 in word2idx and w1 in gensim_model and w2 in gensim_model:
        our_sim = cosine(
            embeddings[word2idx[w1]],
            embeddings[word2idx[w2]]
        )
        gen_sim = cosine(
            gensim_model[w1],
            gensim_model[w2]
        )

        line = f"{w1}-{w2:10s} | {our_sim:.4f} | {gen_sim:.4f}"
        print(line)
        lines.append(line)

with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(lines))

print("\nCosine similarity comparison saved.")
