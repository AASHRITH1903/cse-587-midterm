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

DATASET_DIR = "/scratch/asm6590/cse-587-midterm/dataset"
NLTK_DATA = os.path.join(DATASET_DIR, "nltk_data")

nltk.data.path = [NLTK_DATA]

from nltk.corpus import reuters

# nltk.download("reuters", download_dir=NLTK_DATA)
# nltk.download("punkt", download_dir=NLTK_DATA)
# nltk.download('punkt_tab', download_dir=NLTK_DATA)

categories = reuters.categories()
category_to_index = {category: idx for idx, category in enumerate(categories)}

train_docs = [fid for fid in reuters.fileids() if 'training' in fid]
test_docs = [fid for fid in reuters.fileids() if 'test' in fid]

def tokenize(text):
    return nltk.word_tokenize(text.lower())


# Load / Build vocabulary from training data

VOCAB_FILE = os.path.join(DATASET_DIR, "reuters_vocab.json")

if not os.path.exists(VOCAB_FILE):
    
    print("Building vocab... ")
    
    all_tokens = []
    for doc_id in train_docs:
        all_tokens.extend(tokenize(reuters.raw(doc_id)))

    word_counts = Counter(all_tokens)
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.items() if count > 2]  # Ignore rare words
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    json.dump(word_to_idx, open(VOCAB_FILE, "w"))
    

print("Loading vocab... ")
word_to_idx = json.load(open(VOCAB_FILE))


class ReutersDataset(Dataset):
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        tokens = tokenize(reuters.raw(doc_id))
        indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in tokens]
        label = category_to_index[reuters.categories(doc_id)[0]]  # Assign first category
        return torch.tensor(indices), torch.tensor(label)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_length = max(lengths)

    padded_sequences = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq

    return padded_sequences, torch.tensor(labels)



def get_loader(split='train'):
    
    if split == 'train':
        train_dataset = ReutersDataset(train_docs)
        return DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    elif split == 'test':
        test_dataset = ReutersDataset(test_docs)
        return DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        



# # Load Pretrained Word Embeddings (GloVe)
# glove_vectors = GloVe(name="6B", dim=100)  # Use 100D GloVe embeddings
# embedding_matrix = torch.zeros((len(vocab), 100))

# for word, idx in word_to_idx.items():
#     embedding_matrix[idx] = glove_vectors[word] if word in glove_vectors.stoi else torch.zeros(100)