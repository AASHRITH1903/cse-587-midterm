import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.document_classifier import DocumentClassifier

DATASET = "brown"

if DATASET == "reuters":
    from data.preprocessed.reuters import CONFIG_FILE, EMBED_FILE
    from data.reuters_dataset import get_loader
    
elif DATASET == "movie_reviews":
    from data.preprocessed.movie_reviews import CONFIG_FILE, EMBED_FILE
    from data.movie_reviews_dataset import get_loader
    
elif DATASET == "brown":
    from data.preprocessed.brown import CONFIG_FILE, EMBED_FILE
    from data.brown_dataset import get_loader

config = json.load(open(CONFIG_FILE))

# Model config
INPUT_DIM = config["embed_dim"]
HIDDEN_DIM = config["embed_dim"]
LAYERS = 1
BIDIRECTIONAL = True
CLASSES = len(config["category_to_index"])
VOCAB_SIZE = config["vocab_size"]

EXP_NAME = f"{DATASET}_{LAYERS}_{'bi' if BIDIRECTIONAL else 'uni'}"

# Train config 
EPOCHS = 50
EPOCH_GAP = 1
LOCAL_BATCH_SIZE = 32
LR = 0.001


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


def save_checkpoint(epoch, model, optimizer, scheduler, val_acc, train_acc, train_loss):
    
    checkpoint_path = os.path.join("checkpoints", EXP_NAME, f"checkpoint_epoch_{epoch}.pt")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    data = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),  # Use .module to unwrap DDP model
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
    }
    
    torch.save(data, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")
    
    log_path = os.path.join("checkpoints", EXP_NAME, f"log.json")

    if not os.path.exists(log_path):
        json.dump(
            {"best": None, "data": []}, 
            open(log_path, "w")
        )
        
    log = json.load(open(log_path))
    
    log_data = {
        "epoch": epoch, 
        "val_acc": val_acc, 
        "train_acc": train_acc,
        "train_loss": train_loss
    }
    
    log["data"].append(log_data)  
    
    if (log["best"] == None) or (val_acc > log["best"]["val_acc"]):
        
        log["best"] = log_data
    
    json.dump(log, open(log_path, "w"))
    
            
        
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, map_location):
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.module.load_state_dict(checkpoint["model_state_dict"])  # Use .module for DDP
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}, Resuming from Epoch {checkpoint['epoch']+1}")
        return checkpoint["epoch"]+1
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
        return 1  # Start from scratch if no checkpoint


def evaluate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    # Convert to tensor for all-reduce operation
    total_correct = torch.tensor(total_correct, device=device)
    total_samples = torch.tensor(total_samples, device=device)

    dist.barrier()
    # Synchronize results across all GPUs using all-reduce
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    # Compute final accuracy only on rank 0
    if dist.get_rank() == 0:
        accuracy = total_correct.item() / total_samples.item()
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
    else:
        return None



def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    print(f"Hello from process {rank} using {device}")
    
    # Define model
    model = DocumentClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=LAYERS,
        num_classes=CLASSES,
        vocab_size=VOCAB_SIZE,
        pretrained_emb=torch.load(EMBED_FILE),
        bidirectional=BIDIRECTIONAL,
    )
    model.to(device)
    model = DDP(model, device_ids=[rank])
    
    train_loader = get_loader(rank, world_size, local_batch_size=LOCAL_BATCH_SIZE)
    val_loader = get_loader(rank, world_size, local_batch_size=LOCAL_BATCH_SIZE, split="val")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    start_epoch = load_checkpoint(
        model= model, 
        optimizer= optimizer, 
        scheduler= None, 
        checkpoint_path= None, 
        map_location= device
    )
    
    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            # sync happens here
            optimizer.step()
            
            total_loss += loss.item()
            
        dist.barrier()
            
        if epoch % EPOCH_GAP == 0:
            if rank == 0:
                print(f"Evaluating at epoch {epoch}... ")
                
            dist.barrier()
            train_acc = evaluate(model, train_loader, device)
            dist.barrier()
            val_acc = evaluate(model, val_loader, device)
            
            if val_acc != None:
                save_checkpoint(
                    epoch=epoch, model=model, optimizer=optimizer, scheduler=None, 
                    val_acc=val_acc, train_acc=train_acc, train_loss=total_loss
                )
            
    cleanup()

# Running on multiple GPUs using `torch.multiprocessing`
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
