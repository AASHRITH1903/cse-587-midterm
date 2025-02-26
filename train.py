import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import os
import json

from data.preprocessed.reuters import CONFIG_FILE, EMBED_FILE
from data.reuters_dataset import get_loader
from model import Document_Classifier

config = json.load(open(CONFIG_FILE))
num_classes = len(config["category_to_index"])
num_epochs = 10


def setup(rank, world_size):
    """Initialize the distributed training environment"""
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Use an available port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    print(f"Hello from process {rank} using {device}")
    
    # Define model
    model = Document_Classifier(
        vocab_size= config["vocab_size"],
        embed_dim= config["embed_dim"],
        hidden_dim= config["embed_dim"],
        num_classes= num_classes,
        embeds_path= EMBED_FILE
    )
    model.to(device)
    model = DDP(model, device_ids=[rank])

    dataloader = get_loader(rank, world_size, local_batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            # sync happens here
            optimizer.step()
            
            dist.barrier()
            
            if rank == 0:
                # print("\n\n--------------------------------------\n\n")
                print(f"Loss reported from process {rank} is {loss.item()}")
                
    
    cleanup()

# Running on multiple GPUs using `torch.multiprocessing`
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
