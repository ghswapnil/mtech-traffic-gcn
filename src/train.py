import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import argparse
from src.data.metr_la import load_metr_la_data
from src.data.graph import get_symmetric_normalized_adj
from src.models.adapter import SpatialTemporalAdapter
from src.baseline_ttm import masked_mae_torch, masked_rmse_torch, masked_mape_torch

# PyTorch Loss Function version of Masked MAE (for backpropagation)
class MaskedMAELoss(nn.Module):
    def __init__(self, scaler, null_val=0.0):
        super(MaskedMAELoss, self).__init__()
        self.scaler = scaler
        self.null_val = null_val

    def forward(self, preds, labels):
        # Denormalize to actual speed space (mph)
        mean = self.scaler.mean
        std = self.scaler.std
        
        preds_denorm = preds * std + mean
        labels_denorm = labels * std + mean
        
        mask = (labels_denorm > 0.1) & (labels_denorm != self.null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        loss = torch.abs(preds_denorm - labels_denorm)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)


def train_adapter(use_residual=False):
    # 1. Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 20  # 20 epochs for full paper evaluation run
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_adapter.pth")

    # Use CPU because PyTorch Geometric message-passing has massive overhead/bottlenecks on macOS MPS
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Datasets and Graph
    print("\n--- Loading Data ---")
    train_ds, val_ds, test_ds, adj_mx, scaler = load_metr_la_data(train_stride=1, val_stride=1)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("\n--- Loading Graph structure ---")
    edge_index, edge_weight = get_symmetric_normalized_adj()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # 3. Instantiate Model
    print("\n--- Initializing Model ---")
    model = SpatialTemporalAdapter(
        ttm_model_path="ibm-granite/granite-timeseries-ttm-r2",
        ttm_frozen=True,
        use_residual=use_residual
    )
    model.to(device)

    # 4. Optimizer and Loss Function
    # We train only the parameters where requires_grad=True (GCN + MLP)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = MaskedMAELoss(scaler)

    best_val_loss = float('inf')

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (preds shape: batch_size, 96, 207)
            preds = model(x, edge_index, edge_weight)
            
            # Compute loss on standardized values
            loss = criterion(preds, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x, edge_index, edge_weight)
                loss = criterion(preds, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [+] Validation loss improved. Checkpoint saved to {checkpoint_path}")

    # 6. Evaluation on Test Set
    print("\n--- Loading Best Model for Evaluation ---")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running evaluation on test split...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Model"):
            x = x.to(device)
            preds = model(x, edge_index, edge_weight)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
            
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Invert normalization for real-world metrics
    print("\nInverting normalization...")
    all_preds_denorm = scaler.inverse_transform(all_preds)
    all_targets_denorm = scaler.inverse_transform(all_targets)
    
    # Convert to PyTorch tensors for metric computations
    preds_tensor = torch.tensor(all_preds_denorm, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets_denorm, dtype=torch.float32)
    
    # Horizons setup
    horizons = {
        "15 minutes (Horizon 3)": 2,
        "30 minutes (Horizon 6)": 5,
        "60 minutes (Horizon 12)": 11
    }
    
    print("\n" + "="*50)
    print("          TTM + GCN ADAPTER TEST RESULTS")
    print("="*50)
    
    mae_all = masked_mae_torch(preds_tensor, targets_tensor)
    rmse_all = masked_rmse_torch(preds_tensor, targets_tensor)
    mape_all = masked_mape_torch(preds_tensor, targets_tensor)
    print(f"Overall (Average of 96 steps):")
    print(f"  MAE:  {mae_all:.4f}")
    print(f"  RMSE: {rmse_all:.4f}")
    print(f"  MAPE: {mape_all:.4%}")
    print("-"*50)
    
    for name, step_idx in horizons.items():
        mae_h = masked_mae_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        rmse_h = masked_rmse_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        mape_h = masked_mape_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        print(f"{name}:")
        print(f"  MAE:  {mae_h:.4f}")
        print(f"  RMSE: {rmse_h:.4f}")
        print(f"  MAPE: {mape_h:.4%}")
        print("-"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal Adapter")
    parser.add_argument("--use-residual", action="store_true", help="Use residual connection in fusion MLP")
    args = parser.parse_args()
    
    train_adapter(use_residual=args.use_residual)
