import torch 
from torch.utils.data import Dataset
import numpy as np
import os 

class TextDataset(Dataset):
    def __init__(self, data_folder, block_size=2048):
        self.block_size = block_size
        
        bin_path = os.path.join(data_folder, "data.bin")

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} was not found...")
        
        self.dtype = np.uint16
        self.data = np.memmap(bin_path, dtype=self.dtype, mode='r')
        
        print(f"Dataset was load : {len(self.data):,} tokens.")

    def __len__(self):
        return (len(self.data) - self.block_size - 1) // self.block_size
    
    def __getitem__(self, index):
            start_idx = index * self.block_size
            chunk = self.data[start_idx : start_idx + self.block_size + 1]
            
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
            
            return x, y


class SFTDataset(Dataset):
    def __init__(self, data_path, labels_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.labels = np.memmap(labels_path, dtype=np.int32, mode='r')
        self.block_size = block_size
        self.seq_len = block_size + 1 
        assert len(self.data) == len(self.labels), "SFT data and labels have differents sizes !"

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        
        chunk_data = self.data[i : i + self.seq_len]
        chunk_labels = self.labels[i : i + self.seq_len]
        
        x = torch.from_numpy(chunk_data[:-1].astype(np.int64))
        y = torch.from_numpy(chunk_labels[1:].astype(np.int64)) 
        
        return x, y
