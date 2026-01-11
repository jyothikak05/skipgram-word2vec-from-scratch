import sys
import os
import torch

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.model import SkipGramNS

model = SkipGramNS(vocab_size=10000, embedding_dim=100)

center = torch.randint(0, 10000, (32,))
pos = torch.randint(0, 10000, (32,))
neg = torch.randint(0, 10000, (32, 5))

loss = model(center, pos, neg)
print("Loss:", loss.item())
