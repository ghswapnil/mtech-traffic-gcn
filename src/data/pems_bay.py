"""
PEMS-BAY Dataset Loader.

Loads the PEMS-BAY traffic speed dataset (325 sensors, 6 months).
Starts from Jan 1, 2017.
"""
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from src.data.metr_la import StandardScaler


class PemsBayDataset(Dataset):
    """
    Custom PyTorch Dataset for Spatio-Temporal traffic forecasting on PEMS-BAY.
    Creates sliding windows of size context_len (X) and target_len (y) with stride.
    Optionally returns time features (hour_of_day, day_of_week) for time-conditioned models.
    """
    def __init__(self, data, context_len=512, target_len=96, stride=1,
                 return_time=False, global_start_idx=0):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.context_len = context_len
        self.target_len = target_len
        self.stride = stride
        self.return_time = return_time
        self.global_start_idx = global_start_idx
        
        self.num_samples = (len(data) - context_len - target_len) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x_end = start_idx + self.context_len
        y_end = x_end + self.target_len
        
        x = self.data[start_idx:x_end]
        y = self.data[x_end:y_end]
        
        if self.return_time:
            global_idx = self.global_start_idx + x_end - 1
            total_minutes = global_idx * 5
            hour_of_day = (total_minutes // 60) % 24
            
            # PEMS-BAY starts on Jan 1, 2017 (Sunday -> day_of_week = 6)
            total_days = total_minutes // (24 * 60)
            day_of_week = (6 + total_days) % 7 
            
            time_features = torch.tensor([hour_of_day, day_of_week], dtype=torch.long)
            return x, y, time_features
        
        return x, y


def load_pems_bay_data(datasets_dir="./datasets", context_len=512, target_len=96,
                       train_stride=1, val_stride=1, test_stride=1, return_time=False):
    """
    Loads PEMS-BAY traffic speed data and adjacency matrix.
    """
    h5_path = os.path.join(datasets_dir, "pems-bay.h5")
    adj_path = os.path.join(datasets_dir, "adj_mx_bay.pkl")
    
    if not os.path.exists(h5_path) or not os.path.exists(adj_path):
        raise FileNotFoundError(
            f"Could not find pems-bay.h5 or adj_mx_bay.pkl in {os.path.abspath(datasets_dir)}. "
            f"Please download them and place them there first."
        )

    import h5py
    print("Loading PEMS-BAY traffic speeds from HDF5...")
    with h5py.File(h5_path, 'r') as f:
        data = f['speed']['block0_values'][:]  # Shape: (52116, 325)

        
    print("Loading PEMS-BAY adjacency matrix from pickle...")
    with open(adj_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
        
    print(f"Data shapes: Speed Matrix {data.shape} | Adjacency Matrix {adj_mx.shape}")

    # Chronological Splits (Train 70% | Val 10% | Test 20%)
    num_samples = len(data)
    train_end = int(num_samples * 0.7)
    val_end = int(num_samples * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Standardize data
    mean = train_data.mean()
    std = train_data.std()
    scaler = StandardScaler(mean, std)

    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    # Datasets
    train_dataset = PemsBayDataset(train_norm, context_len, target_len,
                                   stride=train_stride, return_time=return_time, global_start_idx=0)
    val_dataset = PemsBayDataset(val_norm, context_len, target_len,
                                 stride=val_stride, return_time=return_time, global_start_idx=train_end)
    test_dataset = PemsBayDataset(test_norm, context_len, target_len,
                                  stride=test_stride, return_time=return_time, global_start_idx=val_end)

    return train_dataset, val_dataset, test_dataset, adj_mx, scaler
