import os
import json
import torch
from torch import nn
from torchtext.vocab import GloVe


class DocumentClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers,
                 num_classes, vocab_size, pretrained_emb, 
                 bidirectional=True, freeze_emb=False):
        super(DocumentClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        
        self.embedding = None
        
        if pretrained_emb != None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_emb)
        else:
            self.embedding = nn.Embedding(
                num_embeddings= vocab_size, 
                embedding_dim= input_dim
            )
        
        self.seq_enc = nn.LSTM(
            input_size= input_dim,
            hidden_size= hidden_dim, 
            num_layers= num_layers, 
            bidirectional= bidirectional,
        )
        
        self.fc = nn.Linear((2 if bidirectional else 1)*hidden_dim, num_classes)
        
    def forward(self, input_ids):
        
        # print(input_ids.shape)
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
       