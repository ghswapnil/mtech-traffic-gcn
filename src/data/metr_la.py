import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class StandardScaler:
    """Standardize input by removing the mean and scaling to unit variance."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrafficDataset(Dataset):
    """
    Custom PyTorch Dataset for Spatio-Temporal traffic forecasting.
    Creates sliding windows of size context_len (X) and target_len (y) with stride.
    Optionally returns time features (hour_of_day, day_of_week) for time-conditioned models.
    """
    def __init__(self, data, context_len=512, target_len=96, stride=1,
                 return_time=False, global_start_idx=0):
        # data shape: (num_timestamps, num_nodes)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.context_len = context_len
        self.target_len = target_len
        self.stride = stride
        self.return_time = return_time
        self.global_start_idx = global_start_idx  # offset in the full dataset
        
        # Total number of possible sliding windows with stride
        self.num_samples = (len(data) - context_len - target_len) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # X: context window, shape (context_len, num_nodes)
        # y: target window, shape (target_len, num_nodes)
        start_idx = idx * self.stride
        x_end = start_idx + self.context_len
        y_end = x_end + self.target_len
        
        x = self.data[start_idx:x_end]
        y = self.data[x_end:y_end]
        
        if self.return_time:
            # Derive time features from the last timestamp of the context window
            # METR-LA starts at 2012-03-01 00:00:00 with 5-minute intervals
            global_idx = self.global_start_idx + x_end - 1  # last context timestamp
            # Total minutes from start
            total_minutes = global_idx * 5
            # Hour of day (0-23)
            hour_of_day = (total_minutes // 60) % 24
            # Day of week: March 1, 2012 was a Thursday (day_of_week=3)
            total_days = total_minutes // (24 * 60)
            day_of_week = (3 + total_days) % 7  # 0=Mon, 1=Tue, ..., 6=Sun
            
            time_features = torch.tensor([hour_of_day, day_of_week], dtype=torch.long)
            return x, y, time_features
        
        return x, y


def load_metr_la_data(datasets_dir="./datasets", context_len=512, target_len=96,
                       train_stride=1, val_stride=1, test_stride=1,
                       return_time=False):
    """
    Loads traffic speed speeds and adjacency matrix, normalizes data,
    and returns splits and standardizer with optional strides.
    
    Args:
        return_time: If True, dataset __getitem__ returns (x, y, time_features)
                     where time_features = [hour_of_day, day_of_week].
    """
    h5_path = os.path.join(datasets_dir, "metr-la.h5")
    adj_path = os.path.join(datasets_dir, "adj_mx.pkl")
    
    if not os.path.exists(h5_path) or not os.path.exists(adj_path):
        raise FileNotFoundError(
            f"Could not find metr-la.h5 or adj_mx.pkl in {os.path.abspath(datasets_dir)}. "
            f"Please download them and place them there first."
        )

    import h5py
    # 1. Load traffic speeds using h5py to bypass Python 3.14 PyTables import bugs
    print("Loading traffic speeds from HDF5...")
    with h5py.File(h5_path, 'r') as f:
        data = f['df']['block0_values'][:]  # Shape: (34272, 207)
    # 2. Load adjacency matrix
    print("Loading adjacency matrix from pickle...")
    with open(adj_path, 'rb') as f:
        # adj_mx.pkl contains a tuple: (sensor_ids, sensor_id_to_ind, adj_matrix)
        _, _, adj_mx = pickle.load(f, encoding='latin1')
        
    print(f"Data shapes: Speed Matrix {data.shape} | Adjacency Matrix {adj_mx.shape}")

    # 3. Chronological Splits (Train 70% | Val 10% | Test 20%)
    num_samples = len(data)
    train_end = int(num_samples * 0.7)
    val_end = int(num_samples * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 4. Standardize data (using parameters computed strictly from the training split)
    mean = train_data.mean()
    std = train_data.std()
    scaler = StandardScaler(mean, std)

    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    # 5. Create PyTorch Datasets (with optional time features)
    train_dataset = TrafficDataset(train_norm, context_len, target_len,
                                   stride=train_stride, return_time=return_time,
                                   global_start_idx=0)
    val_dataset = TrafficDataset(val_norm, context_len, target_len,
                                 stride=val_stride, return_time=return_time,
                                 global_start_idx=train_end)
    test_dataset = TrafficDataset(test_norm, context_len, target_len,
                                  stride=test_stride, return_time=return_time,
                                  global_start_idx=val_end)

    return train_dataset, val_dataset, test_dataset, adj_mx, scaler


if __name__ == "__main__":
    # Test script to verify local paths
    try:
        train_ds, val_ds, test_ds, adj_mx, scaler = load_metr_la_data()
        print(f"\nSuccessfully loaded datasets:")
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples:   {len(val_ds)}")
        print(f"Test samples:  {len(test_ds)}")
        print(f"Mean speed:    {scaler.mean:.2f} | Std speed: {scaler.std:.2f}")
    except FileNotFoundError as e:
        print("\n[!] Setup required:")
        print(e)
