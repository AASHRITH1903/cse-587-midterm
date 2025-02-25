import os
import json
import torch
from torch import nn
from torchtext.vocab import GloVe



VOCAB_FILE = os.path.join(DATASET_DIR, "reuters_vocab.json")

word_to_idx = json.load(open(VOCAB_FILE))

VOCAB_SIZE = len(word_to_idx)

EMBED_DIM = 100




class DocumentClf(nn.Module):
    
    def __init__(self, dataset_name):
        super(DocumentClf, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embeddings
        embedding_matrix = torch.zeros((vocab_size, embed_dim))

        for word, idx in word_to_idx.items():
            embedding_matrix[idx] = glove_vectors[word] if word in glove_vectors.stoi else torch.zeros(embed_dim)
    
        self.embedding = nn.Embedding(vocab_size, embed_dim, _weight=embedding_matrix)
        
        # LSTMs
        self.fc = nn.Linear(EMBED_DIM, num_classes)
        
    def forward(self, input_ids):
        
        # input_ids: (batch_size, sequence_length)
        print("Input: ", input_ids.shape)
        
        embeds = self.embedding(input_ids)  # (batch_size, sequence_length, embed_dim)
        
        print("Embeds: ", embeds.shape)
        # For simplicity, take mean over sequence_length
        pooled = embeds.mean(dim=1)
        logits = self.fc(pooled)
        return logits