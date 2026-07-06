import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGCN(nn.Module):
    """
    Graph Convolutional Network (GCN) branch for processing spatial relationships 
    between highway traffic sensors.
    """
    def __init__(self, in_channels=512, hidden_dim=64, out_channels=64):
        super(TrafficGCN, self).__init__()
        # First GCN Layer
        self.conv1 = GCNConv(in_channels, hidden_dim)
        # Second GCN Layer
        self.conv2 = GCNConv(hidden_dim, out_channels)
        
    def _batch_graph_index(self, edge_index, edge_weight, batch_size, num_nodes):
        """
        Replicates the static graph connectivity across a batch of samples.
        Creates a disjoint union of B identical graphs for PyG compatibility.
        """
        device = edge_index.device
        
        # 1. Replicate edge_index by adding node offsets
        edge_indices = []
        for b in range(batch_size):
            edge_indices.append(edge_index + b * num_nodes)
        batch_edge_index = torch.cat(edge_indices, dim=1)
        
        # 2. Replicate edge weights
        batch_edge_weight = edge_weight.repeat(batch_size)
        
        return batch_edge_index, batch_edge_weight

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x (torch.Tensor): Node features of shape (batch_size, num_nodes, in_channels)
            edge_index (torch.LongTensor): Graph structure (2, num_edges)
            edge_weight (torch.FloatTensor): Normalized weights (num_edges,)
            
        Returns:
            out (torch.Tensor): Spatial features of shape (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, in_channels = x.shape
        
        # 1. Flatten the batch dimension into the node dimension
        # x_flat shape: (batch_size * num_nodes, in_channels)
        x_flat = x.reshape(batch_size * num_nodes, in_channels)
        
        # 2. Build batched graph index
        batch_edge_index, batch_edge_weight = self._batch_graph_index(
            edge_index, edge_weight, batch_size, num_nodes
        )
        
        # 3. Apply first convolution + ReLU
        h = self.conv1(x_flat, batch_edge_index, batch_edge_weight)
        h = F.relu(h)
        
        # 4. Apply second convolution
        out_flat = self.conv2(h, batch_edge_index, batch_edge_weight)
        
        # 5. Reshape back to (batch_size, num_nodes, out_channels)
        out = out_flat.reshape(batch_size, num_nodes, -1)
        return out


if __name__ == "__main__":
    # Test block to verify GCN layers and batch processing
    print("Testing TrafficGCN module...")
    
    # Mock data: Batch Size = 4, Nodes = 207, History features = 512
    x = torch.randn(4, 207, 512)
    
    # Mock sparse graph: Node 0 connected to 1 and self-loops
    edge_index = torch.tensor([[0, 1, 0, 1],
                               [1, 0, 0, 1]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    
    # Instantiate GCN
    gcn = TrafficGCN(in_channels=512, hidden_dim=64, out_channels=64)
    
    # Forward pass
    out = gcn(x, edge_index, edge_weight)
    
    print("\nVerification successful!")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in gcn.parameters())}")
