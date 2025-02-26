import os 
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from collections import Counter
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import nltk

DATA_DIR = "/scratch/asm6590/cse-587-midterm/data"
NLTK_DATA = os.path.join(DATA_DIR, "nltk_data")
nltk.data.path = [NLTK_DATA]

from nltk.corpus import reuters

from data.preprocessed.reuters import (
    CONFIG_FILE, 
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



def get_loader(rank, world_size, local_batch_size=32, split='train'):
    
    if split == 'train':
        dataset = ReutersDataset(doc_ids= config["train_docs"])
        sampler = DistributedSampler(
            dataset= dataset, 
            num_replicas= world_size, 
            rank= rank, 
            shuffle= True
        )
        loader = DataLoader(
            dataset= dataset, 
            batch_size= local_batch_size, 
            sampler= sampler, 
            collate_fn= collate_fn, 
            num_workers= 4
        )
        return loader
    
    elif split == 'test':
        pass
    
        