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

from data.preprocessed.reuters import (
    CONFIG_FILE, 
    EMBED_FILE,
    tokenize
)

config = json.load(open(CONFIG_FILE))

word_to_idx = config["word_to_idx"]
category_to_index = config["category_to_index"]


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



def reuters_loader(split='train'):
    
    if split == 'train':
        train_dataset = ReutersDataset(config["train_docs"])
        return DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    elif split == 'test':
        test_dataset = ReutersDataset(config["test_docs"])
        return DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        