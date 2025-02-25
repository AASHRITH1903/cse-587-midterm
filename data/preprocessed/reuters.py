import os 
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import nltk

DATA_DIR = "/scratch/asm6590/cse-587-midterm/data"
NLTK_DATA = os.path.join(DATA_DIR, "nltk_data")

nltk.data.path = [NLTK_DATA]

from nltk.corpus import reuters

nltk.download("reuters", download_dir=NLTK_DATA)
nltk.download("punkt", download_dir=NLTK_DATA)
nltk.download('punkt_tab', download_dir=NLTK_DATA)


def tokenize(text):
    return nltk.word_tokenize(text.lower())

GLOVE_CACHE = os.path.join(DATA_DIR, "glove_cache")
CONFIG_FILE = os.path.join(DATA_DIR, "preprocessed", "reuters_config.json")
EMBED_FILE = os.path.join(DATA_DIR, "preprocessed", "reuters_embeddings.pt")


if __name__ == "__main__":
    
    # Build vocabulary from training data
    
    train_docs = [fid for fid in reuters.fileids() if 'training' in fid]
    test_docs = [fid for fid in reuters.fileids() if 'test' in fid]
        
    categories = reuters.categories()
    category_to_index = {category: idx for idx, category in enumerate(categories)}

    print("Building vocab... ")
    all_tokens = []
    for doc_id in train_docs:
        all_tokens.extend(tokenize(reuters.raw(doc_id)))

    word_counts = Counter(all_tokens)
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.items() if count > 2]  # Ignore rare words
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    vocab_size = len(vocab)
    embed_dim = 100


    # Embeddings
    print("Building Embeddings... ")
    glove_vectors = GloVe(name="6B", dim=embed_dim, cache=GLOVE_CACHE)
    embedding_matrix = torch.zeros((vocab_size, embed_dim))

    for word, idx in word_to_idx.items():
        embedding_matrix[idx] = glove_vectors[word] if word in glove_vectors.stoi else torch.zeros(embed_dim)
        
    torch.save(embedding_matrix, EMBED_FILE)
    
    config = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "train_docs": train_docs,
        "test_docs": test_docs,
        "category_to_index": category_to_index,
        "vocab": vocab,
        "word_to_idx": word_to_idx,
    }
    
    json.dump(config, open(CONFIG_FILE, "w"))
        