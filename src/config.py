import torch

config = {

    # TARGETS TOKEN FOR PRETRAINING OT SFT
    'stf_target_tokens': 100_000_000,
    'pre_training_target_tokens': 20_000_000_000,

    # PATHS
    'dataset_path': "data/raw",
    'tokenizer_path': "data/tokenizer",
    'checkpoint_file_path': "checkpoints/checkpoint.pt",

    # MODEL ARCHITECTURE
    'd_model': 576,
    'n_layer': 30,
    'n_head': 9,
    'block_size': 2048,
    'n_kv_head': 3,
    'vocab_size': 49152,

    # LayerNorm 
    'use_layernorm': True, # Using LayerNorm instead of RMSNorm

    # BATCHS
    'batch_size': 256,
    'micro_batch_size': 4,  
    
    # OPTIMIZER
    'max_lr': 3e-3,
    
    # LOGGING INTERVAL
    'logging_interval': 2,

    # DEVICE
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    
}