# Document Classification with Distributed Training

This project implements document classification using PyTorch with distributed data parallel (DDP) training across multiple GPUs. The model is designed to classify text documents from different datasets.

## Datasets Supported
- **Reuters**
- **Movie Reviews**
- **Brown Corpus**

## Prerequisites
### Hardware
- Multi-GPU setup with NVIDIA GPUs
- CUDA installed

### Software
- Python 3.8+
- PyTorch 2.0+
- NVIDIA NCCL
- CUDA Toolkit

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Setup
Make sure the dataset is preprocessed and stored under `data/preprocessed/{dataset_name}`.

Each dataset folder should have:
- `config.json`: Configuration file with the following structure:
  ```json
  {
    "embed_dim": 300,
    "vocab_size": 10000,
    "category_to_index": {"category1": 0, "category2": 1}
  }
  ```
- `embeddings.pt`: Pre-trained word embeddings as a torch tensor.

## Configuration
Modify the `DATASET` variable in `train.py` to choose the dataset:
```python
DATASET = "brown"  # Options: "reuters", "movie_reviews", "brown"
```

### Model Parameters
Set model configuration parameters in `config.json`:
| Parameter     | Description             |
|--------------|-----------------------|
| embed_dim    | Embedding dimension   |
| vocab_size   | Vocabulary size      |
| category_to_index | Class labels mapping |

### Training Parameters
In the `train.py` script:
```python
EPOCHS = 50
LOCAL_BATCH_SIZE = 32
LR = 0.001
LAYERS = 1
BIDIRECTIONAL = False
```

## Run Training
### Single Node Multi-GPU Training
To launch training across all available GPUs:
```bash
python train.py
```

### Checkpoints
Checkpoints are stored in the `checkpoints/{EXP_NAME}` folder, where `EXP_NAME` is generated based on dataset and model configurations.

Logs are saved in `log.json` containing:
- Epoch
- Validation Accuracy
- Training Accuracy
- Training Loss

## Resuming from Checkpoint
Modify the `checkpoint_path` argument in the `load_checkpoint` function to specify the checkpoint to resume from.

## Evaluation
After training, the final model's accuracy is printed automatically at each evaluation step.

## Notes
- Ensure that `NCCL` backend is properly set up for distributed training.
- Use `torch.cuda.device_count()` to verify the number of available GPUs.

