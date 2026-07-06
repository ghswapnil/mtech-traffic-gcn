import torch
import torch.nn as nn
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from src.models.gcn import TrafficGCN

class SpatialTemporalAdapter(nn.Module):
    """
    Spatio-Temporal GNN Adapter.
    Fuses a frozen temporal foundation model (TTM) with a trainable GCN spatial branch.
    """
    def __init__(self, ttm_model_path="ibm-granite/granite-timeseries-ttm-r2", 
                 ttm_frozen=True, gcn_hidden_dim=64, gcn_out_channels=64, 
                 context_len=512, forecast_len=96, use_residual=False):
        super(SpatialTemporalAdapter, self).__init__()
        self.use_residual = use_residual
        
        # 1. Load TTM (Temporal Branch)
        print(f"Initializing TTM backbone from {ttm_model_path}...")
        self.ttm = TinyTimeMixerForPrediction.from_pretrained(ttm_model_path)
        
        if ttm_frozen:
            print("Freezing TTM parameters...")
            for param in self.ttm.parameters():
                param.requires_grad = False
        self.ttm_frozen = ttm_frozen
        
        # 2. Initialize GCN (Spatial Branch)
        print(f"Initializing GCN spatial branch (hidden={gcn_hidden_dim}, out={gcn_out_channels})...")
        self.gcn = TrafficGCN(in_channels=context_len, hidden_dim=gcn_hidden_dim, out_channels=gcn_out_channels)
        
        # 3. Initialize MLP Fusion Head
        # Concatenated feature size: TTM prediction length (forecast_len) + GCN features (gcn_out_channels)
        in_features = forecast_len + gcn_out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features // 2, forecast_len)
        )
        print(f"MLP Fusion initialized: input dimension = {in_features} -> output dimension = {forecast_len}")

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x (torch.Tensor): Input speed windows, shape (batch_size, context_len=512, num_nodes=207)
            edge_index (torch.LongTensor): Graph structure indices (2, num_edges)
            edge_weight (torch.FloatTensor): Symmetrically normalized weights (num_edges,)
            
        Returns:
            final_forecast (torch.Tensor): Spatio-temporal predictions, shape (batch_size, forecast_len=96, num_nodes=207)
        """
        batch_size, context_len, num_nodes = x.shape
        
        # 1. Temporal Branch: Zero-Shot prediction using TTM
        # past_values shape: (batch_size, context_len, num_nodes)
        ttm_out = self.ttm(past_values=x)
        # ttm_preds shape: (batch_size, forecast_len, num_nodes)
        ttm_preds = ttm_out.prediction_outputs
        
        # Transpose to (batch_size, num_nodes, forecast_len) for node-level concatenation
        ttm_preds_trans = ttm_preds.transpose(1, 2)
        
        # 2. Spatial Branch: GCN message passing
        # GCN expects input shape: (batch_size, num_nodes, context_len)
        # We pass x transposed to match node-first shape
        x_trans = x.transpose(1, 2) # Shape: (batch_size, num_nodes, context_len)
        spatial_feats = self.gcn(x_trans, edge_index, edge_weight) # Shape: (batch_size, num_nodes, gcn_out_channels)
        
        # 3. Feature Fusion: Concatenate temporal forecasts and spatial features
        # combined shape: (batch_size, num_nodes, forecast_len + gcn_out_channels)
        combined = torch.cat([ttm_preds_trans, spatial_feats], dim=-1)
        
        # 4. Final Projection: Predict corrected forecasts per node
        # fused shape: (batch_size, num_nodes, forecast_len)
        if self.use_residual:
            fused = ttm_preds_trans + self.mlp(combined)
        else:
            fused = self.mlp(combined)
        
        # Transpose back to standard traffic shape: (batch_size, forecast_len, num_nodes)
        final_forecast = fused.transpose(1, 2)
        return final_forecast


if __name__ == "__main__":
    # Test block to verify full SpatialTemporalAdapter forward pass
    print("Testing SpatialTemporalAdapter module...")
    
    # Mock data: Batch Size = 2, Context = 512, Nodes = 207
    x = torch.randn(2, 512, 207)
    
    # Mock graph: simple fully connected list of 207 nodes
    edge_index = torch.zeros((2, 207), dtype=torch.long)
    edge_index[0] = torch.arange(207)
    edge_index[1] = torch.arange(207)
    edge_weight = torch.ones(207, dtype=torch.float32)
    
    # Instantiate Adapter
    model = SpatialTemporalAdapter(ttm_model_path="ibm-granite/granite-timeseries-ttm-r2", ttm_frozen=True)
    
    # Forward pass
    out = model(x, edge_index, edge_weight)
    
    print("\nVerification successful!")
    print(f"Input shape:          {x.shape}")
    print(f"Output shape:         {out.shape}")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total Parameters:     {total_params}")
    print(f"Trainable Parameters: {trainable_params} ({trainable_params / total_params:.2%})")
    print(f"Frozen Parameters:    {frozen_params}")
