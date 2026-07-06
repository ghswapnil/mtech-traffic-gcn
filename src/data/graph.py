import os
import pickle
import numpy as np
import torch

def get_symmetric_normalized_adj(adj_path="./datasets/adj_mx.pkl"):
    """
    Loads adj_mx.pkl and computes the symmetric GCN normalized adjacency matrix:
    A_hat = D^{-1/2} * (A + I) * D^{-1/2}
    
    Returns:
        edge_index (torch.LongTensor): Shape (2, num_edges) in COO format
        edge_weight (torch.FloatTensor): Shape (num_edges,) with normalized values
    """
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Could not find adjacency matrix at: {os.path.abspath(adj_path)}")
        
    # 1. Load the pickle file
    print(f"Loading road network graph from {adj_path}...")
    with open(adj_path, 'rb') as f:
        # pickle contains: (sensor_ids, sensor_id_to_ind, adj_matrix)
        _, _, adj_mx = pickle.load(f, encoding='latin1')
        
    num_nodes = adj_mx.shape[0]
    print(f"Loaded graph with {num_nodes} nodes.")
    
    # 2. Add self-loops (A_tilde = A + I)
    adj_tilde = adj_mx + np.eye(num_nodes)
    
    # 3. Calculate Degree matrix D_tilde and invert square root
    # D_tilde[i, i] = sum_j (A_tilde[i, j])
    d = np.sum(adj_tilde, axis=1)
    
    # Avoid division by zero
    d_inv_sqrt = np.power(d, -0.5, where=(d > 0))
    d_inv_sqrt[d <= 0] = 0.0
    
    # Create diagonal D^{-1/2} matrix
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Compute A_hat = D^{-1/2} * A_tilde * D^{-1/2}
    adj_normalized = d_mat_inv_sqrt.dot(adj_tilde).dot(d_mat_inv_sqrt)
    
    # 4. Convert to PyTorch sparse representation (COO format)
    adj_tensor = torch.tensor(adj_normalized, dtype=torch.float32)
    adj_sparse = adj_tensor.to_sparse()
    
    edge_index = adj_sparse.indices().long()
    edge_weight = adj_sparse.values().float()
    
    print(f"Graph normalized: {edge_index.shape[1]} active edges (including self-loops).")
    return edge_index, edge_weight


if __name__ == "__main__":
    # Test normalization function
    try:
        edge_index, edge_weight = get_symmetric_normalized_adj()
        print("\nNormalization Successful!")
        print(f"Edge Index shape:  {edge_index.shape}")
        print(f"Edge Weight shape: {edge_weight.shape}")
        print("\nFirst 5 normalized connections:")
        for i in range(5):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            weight = edge_weight[i].item()
            print(f"  Sensor {src} -> Sensor {dst} | Weight: {weight:.4f}")
    except Exception as e:
        print(f"\n[!] Error: {e}")
