"""
Time-Conditioned GCN Forecaster.

A standalone spatial forecasting model where GCN edge weights are
dynamically modulated by time-of-day and day-of-week embeddings.
This is Stage 1 of the two-stage residual boosting architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TimeEmbedding(nn.Module):
    """
    Embeds hour_of_day (0-23) and day_of_week (0-6) into dense vectors.
    These learned embeddings capture temporal patterns like rush hours
    and weekday/weekend differences.
    """
    def __init__(self, hour_dim=16, day_dim=8):
        super(TimeEmbedding, self).__init__()
        self.hour_embed = nn.Embedding(24, hour_dim)   # 24 hours
        self.day_embed = nn.Embedding(7, day_dim)      # 7 days
        self.out_dim = hour_dim + day_dim
    
    def forward(self, time_features):
        """
        Args:
            time_features: (batch_size, 2) where [:, 0] = hour, [:, 1] = day_of_week
        Returns:
            time_emb: (batch_size, hour_dim + day_dim)
        """
        hour = time_features[:, 0]  # (batch_size,)
        day = time_features[:, 1]   # (batch_size,)
        
        hour_emb = self.hour_embed(hour)  # (batch_size, hour_dim)
        day_emb = self.day_embed(day)      # (batch_size, day_dim)
        
        return torch.cat([hour_emb, day_emb], dim=-1)  # (batch_size, hour_dim + day_dim)


class DynamicAdjacency(nn.Module):
    """
    Modulates static edge weights based on time embeddings.
    
    At rush hour (e.g., Monday 8 AM), highway on-ramp connections become
    stronger. At midnight, sensors become nearly independent. The model
    learns these patterns automatically from data.
    
    A_dynamic = A_static * sigmoid(MLP(time_embedding))
    """
    def __init__(self, time_emb_dim, hidden_dim=32):
        super(DynamicAdjacency, self).__init__()
        # MLP that maps time embedding to a scalar modulation factor
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # output between 0 and 1
        )
    
    def forward(self, edge_weight, time_emb):
        """
        Args:
            edge_weight: (num_edges,) static normalized edge weights
            time_emb: (batch_size, time_emb_dim) time embedding vector
        Returns:
            dynamic_edge_weight: (batch_size, num_edges) time-modulated weights
        """
        # Compute modulation factor from time embedding
        # modulation: (batch_size, 1)
        modulation = self.mlp(time_emb)  # (batch_size, 1)
        
        # Scale static weights: each batch sample gets its own modulated graph
        # edge_weight: (num_edges,) -> broadcast with modulation: (batch_size, 1)
        # Result: (batch_size, num_edges)
        dynamic_weights = edge_weight.unsqueeze(0) * modulation
        
        return dynamic_weights


class TimeConditionedGCN(nn.Module):
    """
    Two-layer GCN with time-conditioned edge weights.
    Uses the same GCNConv architecture as TrafficGCN but with dynamic adjacency.
    """
    def __init__(self, in_channels=512, hidden_dim=128, out_channels=128):
        super(TimeConditionedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: (batch_size * num_nodes, in_channels) flattened node features
            edge_index: (2, batch_size * num_edges) batched edge indices
            edge_weight: (batch_size * num_edges,) batched edge weights
        Returns:
            out: (batch_size * num_nodes, out_channels) spatial features
        """
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        out = self.conv2(h, edge_index, edge_weight)
        return out


class TimeConditionedGCNForecaster(nn.Module):
    """
    Full Stage 1 model: Time-Conditioned GCN that produces 96-step traffic forecasts.
    
    Architecture:
        1. TimeEmbedding: hour + day_of_week -> dense vector
        2. DynamicAdjacency: modulates road network weights by time
        3. TimeConditionedGCN: spatial message passing with dynamic edges
        4. Output MLP: projects spatial features + time embedding to 96 future steps
    """
    def __init__(self, context_len=512, forecast_len=96, num_nodes=207,
                 gcn_hidden=128, gcn_out=128, hour_dim=16, day_dim=8):
        super(TimeConditionedGCNForecaster, self).__init__()
        
        self.num_nodes = num_nodes
        self.forecast_len = forecast_len
        self.context_len = context_len
        
        # 1. Time Embedding
        self.time_embed = TimeEmbedding(hour_dim=hour_dim, day_dim=day_dim)
        time_emb_dim = hour_dim + day_dim  # 24
        
        # 2. Dynamic Adjacency
        self.dynamic_adj = DynamicAdjacency(time_emb_dim=time_emb_dim)
        
        # 3. GCN with dynamic edges
        self.gcn = TimeConditionedGCN(
            in_channels=context_len,
            hidden_dim=gcn_hidden,
            out_channels=gcn_out
        )
        
        # 4. Output MLP: GCN features + time embedding -> forecast
        mlp_input = gcn_out + time_emb_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input, mlp_input),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_input, forecast_len)
        )
        
        print(f"TimeConditionedGCNForecaster initialized:")
        print(f"  GCN: {context_len} -> {gcn_hidden} -> {gcn_out}")
        print(f"  Time embedding: hour({hour_dim}) + day({day_dim}) = {time_emb_dim}")
        print(f"  Output MLP: {mlp_input} -> {forecast_len}")
    
    def _batch_graph(self, edge_index, edge_weight_batch, batch_size, num_nodes):
        """
        Creates a batched disjoint graph for PyG.
        
        Args:
            edge_index: (2, num_edges) static edge indices
            edge_weight_batch: (batch_size, num_edges) time-modulated weights
            batch_size: number of samples
            num_nodes: nodes per graph
        Returns:
            batched_edge_index: (2, batch_size * num_edges)
            batched_edge_weight: (batch_size * num_edges,)
        """
        device = edge_index.device
        num_edges = edge_index.shape[1]
        
        # Replicate edge indices with node offsets
        edge_indices = []
        edge_weights = []
        for b in range(batch_size):
            edge_indices.append(edge_index + b * num_nodes)
            edge_weights.append(edge_weight_batch[b])  # (num_edges,)
        
        batched_edge_index = torch.cat(edge_indices, dim=1)
        batched_edge_weight = torch.cat(edge_weights, dim=0)
        
        return batched_edge_index, batched_edge_weight
    
    def forward(self, x, edge_index, edge_weight, time_features):
        """
        Args:
            x: (batch_size, context_len, num_nodes) input traffic speeds
            edge_index: (2, num_edges) static graph structure
            edge_weight: (num_edges,) static normalized edge weights
            time_features: (batch_size, 2) [hour_of_day, day_of_week]
        Returns:
            forecast: (batch_size, forecast_len, num_nodes) traffic predictions
        """
        batch_size = x.shape[0]
        
        # 1. Compute time embedding
        time_emb = self.time_embed(time_features)  # (batch_size, 24)
        
        # 2. Compute dynamic edge weights
        # dynamic_weights: (batch_size, num_edges)
        dynamic_weights = self.dynamic_adj(edge_weight, time_emb)
        
        # 3. Prepare input for GCN: (batch_size, num_nodes, context_len)
        x_trans = x.transpose(1, 2)
        x_flat = x_trans.reshape(batch_size * self.num_nodes, self.context_len)
        
        # 4. Build batched graph with dynamic weights
        batched_edge_index, batched_edge_weight = self._batch_graph(
            edge_index, dynamic_weights, batch_size, self.num_nodes
        )
        
        # 5. GCN message passing
        gcn_out = self.gcn(x_flat, batched_edge_index, batched_edge_weight)
        # gcn_out: (batch_size * num_nodes, gcn_out_dim)
        gcn_out = gcn_out.reshape(batch_size, self.num_nodes, -1)
        # gcn_out: (batch_size, num_nodes, gcn_out_dim)
        
        # 6. Concatenate GCN features with time embedding (broadcast to all nodes)
        # time_emb: (batch_size, time_emb_dim) -> (batch_size, num_nodes, time_emb_dim)
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)
        combined = torch.cat([gcn_out, time_emb_expanded], dim=-1)
        # combined: (batch_size, num_nodes, gcn_out_dim + time_emb_dim)
        
        # 7. Project to forecast
        forecast = self.output_mlp(combined)
        # forecast: (batch_size, num_nodes, forecast_len)
        
        # Transpose to standard shape: (batch_size, forecast_len, num_nodes)
        forecast = forecast.transpose(1, 2)
        
        return forecast


if __name__ == "__main__":
    print("Testing TimeConditionedGCNForecaster...")
    
    # Mock data
    batch_size = 4
    x = torch.randn(batch_size, 512, 207)
    time_features = torch.tensor([
        [8, 0],   # Monday 8 AM
        [17, 4],  # Friday 5 PM
        [3, 6],   # Sunday 3 AM
        [12, 2],  # Wednesday noon
    ], dtype=torch.long)
    
    # Mock graph
    edge_index = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    
    # Instantiate
    model = TimeConditionedGCNForecaster()
    
    # Forward pass
    out = model(x, edge_index, edge_weight, time_features)
    
    print(f"\nInput shape:       {x.shape}")
    print(f"Time features:     {time_features.shape}")
    print(f"Output shape:      {out.shape}")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:  {total}")
    print(f"Trainable:         {trainable}")
