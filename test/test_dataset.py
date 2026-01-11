import sys
import os

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.dataset import SkipGramDataset

dataset = SkipGramDataset(
    file_path="data/wiki_clean.txt",
    window_size=5,
    min_count=5
)

pairs = dataset.generate_pairs()

print("Total training pairs:", len(pairs))
print("Sample pairs:", pairs[:10])
