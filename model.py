import os
import json
import torch
from torch import nn
from torchtext.vocab import GloVe



class Document_Classifier(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embeds_path):
        super(Document_Classifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(
            num_embeddings= vocab_size, 
            embedding_dim= embed_dim, 
            _weight= torch.load(embeds_path)
        )
        self.seq_enc = nn.LSTM(
            input_size= embed_dim, 
            hidden_size= hidden_dim, 
            num_layers=2, 
            bidirectional=True,
        )
        self.fc = nn.Linear(2*hidden_dim, num_classes)
        
    def forward(self, input_ids):
        
        input_ids = input_ids.permute(1, 0)
        # print("input_ids", input_ids.shape)
        
        embeds = self.embedding(input_ids) 
        # print("embeds", embeds.shape)
        
        encs, _ = self.seq_enc(embeds)
        # print("encs", encs.shape)
        # print("h", h.shape)
        # print("c", c.shape)
        
        doc_enc = torch.mean(encs, dim=0)
        # print("doc_enc", doc_enc.shape)
        
        logits = self.fc(doc_enc)
        # print("logits", logits.shape)
        
        return logits
       
        
        
        