"""
Model 4: TTM Linear Probing
TTM Backbone: FROZEN
TTM Head: TRAINED
GCN: None
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from src.data.metr_la import load_metr_la_data
from src.model1_vanilla_ttm import masked_mae_torch, masked_rmse_torch, masked_mape_torch

class MaskedMAELoss(nn.Module):
    def __init__(self, scaler, null_val=0.0):
        super(MaskedMAELoss, self).__init__()
        self.scaler = scaler
        self.null_val = null_val

    def forward(self, preds, labels):
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

def train_linear_probing(stride=1, epochs=20, batch_size=128, lr=0.001):
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_ttm_linear_probing.pth")
    
    device = torch.device("cpu")
    # Use all available CPU cores for matrix math
    torch.set_num_threads(os.cpu_count())
    print(f"Using device: {device} with {os.cpu_count()} threads")
    
    print("\n--- Loading Data ---")
    train_ds, val_ds, test_ds, _, scaler = load_metr_la_data(train_stride=stride, val_stride=1)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print("\n--- Initializing TTM (Linear Probing Mode) ---")
    model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")
    model.to(device)
    
    # Freeze backbone, unfreeze head
    print("Freezing TTM Backbone...")
    print("Unfreezing TTM Head...")
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total} | Trainable params (Head): {trainable}")
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = MaskedMAELoss(scaler)
    
    best_val_loss = float('inf')
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # TTM expects past_values shape: (batch_size, context_len, num_nodes)
            out = model(past_values=x)
            preds = out.prediction_outputs
            
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(past_values=x)
                preds = out.prediction_outputs
                loss = criterion(preds, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [+] Saved checkpoint: {checkpoint_path}")

    # Evaluation
    print("\n--- Evaluating Best Model ---")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            out = model(past_values=x)
            all_preds.append(out.prediction_outputs.cpu().numpy())
            all_targets.append(y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    all_preds_denorm = scaler.inverse_transform(all_preds)
    all_targets_denorm = scaler.inverse_transform(all_targets)
    
    preds_tensor = torch.tensor(all_preds_denorm, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets_denorm, dtype=torch.float32)
    
    print("\n" + "="*50)
    print("   MODEL 4: TTM LINEAR PROBING TEST RESULTS")
    print("="*50)
    
    mae_all = masked_mae_torch(preds_tensor, targets_tensor)
    rmse_all = masked_rmse_torch(preds_tensor, targets_tensor)
    mape_all = masked_mape_torch(preds_tensor, targets_tensor)
    print(f"Overall MAE: {mae_all:.4f} | RMSE: {rmse_all:.4f} | MAPE: {mape_all:.4%}")
    print("-"*50)
    
    horizons = {"15 min": 2, "30 min": 5, "60 min": 11}
    for name, step_idx in horizons.items():
        mae_h = masked_mae_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        rmse_h = masked_rmse_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        mape_h = masked_mape_torch(preds_tensor[:, step_idx, :], targets_tensor[:, step_idx, :])
        print(f"{name}: MAE: {mae_h:.4f} | RMSE: {rmse_h:.4f} | MAPE: {mape_h:.4%}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_linear_probing(epochs=args.epochs)
