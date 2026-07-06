import torch
import numpy as np
from torch.utils.data import DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from src.data.metr_la import load_metr_la_data

# Masked evaluation metrics (standard in spatio-temporal traffic literature)
def masked_mae_torch(preds, labels, null_val=0.0):
    mask = (labels > 0.1) & (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss).item()

def masked_mape_torch(preds, labels, null_val=0.0):
    mask = (labels > 0.1) & (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss).item()

def masked_rmse_torch(preds, labels, null_val=0.0):
    mask = (labels > 0.1) & (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss)).item()


def run_baseline_evaluation():
    # 1. Load data splits and standardizer
    print("Loading METR-LA test dataset...")
    _, _, test_dataset, _, scaler = load_metr_la_data(datasets_dir="./datasets")
    
    # 2. Setup DataLoader (using batch size of 64 for speed)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. Load TTM Model
    print("Loading pre-trained TTM model from Hugging Face...")
    device = torch.device("cpu")
    print("Using CPU for inference.")
        
    model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")
    model.to(device)
    model.eval()
    
    from tqdm import tqdm
    
    all_preds = []
    all_targets = []
    
    print("Running zero-shot inference on test split...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating TTM"):
            # x shape: (batch_size, context_len=512, num_nodes=207)
            # y shape: (batch_size, target_len=96, num_nodes=207)
            x = x.to(device)
            
            # Forward pass through TTM
            outputs = model(past_values=x)
            preds = outputs.prediction_outputs # Shape: (batch_size, target_len=96, num_nodes=207)
            
            # Move to CPU for metrics
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
                
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0) # Shape: (N, 96, 207)
    all_targets = np.concatenate(all_targets, axis=0) # Shape: (N, 96, 207)
    
    # 4. Denormalize data to return to actual traffic speeds (mph)
    print("\nInverting normalization...")
    all_preds_denorm = scaler.inverse_transform(all_preds)
    all_targets_denorm = scaler.inverse_transform(all_targets)
    
    # Convert back to torch tensors for unified metric functions
    preds_tensor = torch.tensor(all_preds_denorm, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets_denorm, dtype=torch.float32)
    
    # 5. Evaluate overall and specific time horizons:
    # 5-minute intervals -> index 2 is 15-min, index 5 is 30-min, index 11 is 60-min
    horizons = {
        "15 minutes (Horizon 3)": 2,
        "30 minutes (Horizon 6)": 5,
        "60 minutes (Horizon 12)": 11
    }
    
    print("\n" + "="*50)
    print("           VANILLA TTM ZERO-SHOT RESULTS")
    print("="*50)
    
    # Overall metrics across all 96 forecast steps
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
    run_baseline_evaluation()
