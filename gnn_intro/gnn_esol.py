
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
import os


# Resolve datasets folder path relative to this script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(current_dir, "../datasets")
os.makedirs(datasets_dir, exist_ok=True)

# 1. Load the ESOL dataset (solubility data)
print("Loading ESOL dataset...")
dataset = MoleculeNet(root=datasets_dir, name="ESOL")

print(f"Dataset loaded. Length: {len(dataset)}")
print(f"Number of node features: {dataset.num_features}")
print(f"Number of targets: {dataset.num_classes}")

# 2. Define the GCN Model
embedding_size = 64

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN conv layers
        self.initial_conv = GCNConv(dataset.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer (stacking max and mean pool requires embedding_size * 2)
        self.out = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        # First GCN layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # GCN layer 2
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        # GCN layer 3
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        # GCN layer 4
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (combining Global Max Pool and Global Mean Pool)
        pooled = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Final linear prediction
        out = self.out(pooled)
        return out, hidden

model = GCN()
print("\n--- GCN Model Architecture ---")
print(model)
print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")

# 3. Setup device, loaders, loss function, and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  

# Force CPU device locally to avoid MPS overhead and hangs on macOS
device = torch.device("cpu")

print(f"\nTraining on device: {device}")
model = model.to(device)

# Wrap splits into PyG DataLoaders
data_size = len(dataset)
train_loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=False)

def train_epoch():
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)  
        optimizer.zero_grad() 
        
        # Pass node features (float), edge connections, and graph indices
        pred, _ = model(batch.x.float(), batch.edge_index, batch.batch) 
        
        # Calculate loss and step gradients
        loss = loss_fn(pred, batch.y)     
        loss.backward()  
        optimizer.step()
        
        epoch_loss += loss.item() * batch.num_graphs
    return epoch_loss / len(train_loader.dataset)

import matplotlib.pyplot as plt

# 4. Training loop
print("\nStarting training loop...")
losses = []
for epoch in range(1, 501):
    loss = train_epoch()
    losses.append(loss)
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train MSE Loss: {loss:.4f}")

# 5. Evaluate on the Test Set
model.eval()
test_loss = 0
y_real = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y)
        test_loss += loss.item() * batch.num_graphs
        
        y_real.extend(batch.y.cpu().numpy().flatten())
        y_pred.extend(pred.cpu().numpy().flatten())

print(f"\nFinal Test MSE Loss: {test_loss / len(test_loader.dataset):.4f}")

# 6. Save Plots to Files
plots_dir = os.path.join(current_dir, "../plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, 501), losses, color="blue", label="Train Loss")
plt.title("GCN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
plt.close()

# Plot 2: Real vs Predicted Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_real, y_pred, alpha=0.6, color="teal", label="Compounds")
plt.title("ESOL Solubility: Real vs. Predicted")
plt.xlabel("Real Solubility (log mol/L)")
plt.ylabel("Predicted Solubility (log mol/L)")
plt.xlim(-7, 2)
plt.ylim(-7, 2)
plt.plot([-7, 2], [-7, 2], color="red", linestyle="--", label="Identity (y = x)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "predictions_scatter.png"), dpi=150)
plt.close()

print(f"\nVisualizations saved successfully in: {os.path.abspath(plots_dir)}")
