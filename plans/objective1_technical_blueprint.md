# Technical Blueprint: Objective 1 (Spatial Adapter for Frozen TTM)

This document outlines the exact technical implementation plan for Month 3 of the M.Tech project. The goal is to build a spatial adapter that injects graph structure into the frozen IBM Granite TTM backbone.

---

## 1. Data Shapes and Inputs

Before writing the adapter, we must define the mathematical shapes of our inputs. Using a standard traffic dataset like METR-LA:

**Input 1: The Adjacency Matrix (`A`)**
*   **Shape:** `[N, N]` (e.g., `[207, 207]`)
*   **Meaning:** A matrix representing the road network. `A[i, j]` contains the normalized spatial distance/connectivity between Sensor `i` and Sensor `j`.

**Input 2: The Time-Series Features (`X`)**
*   **Shape:** `[Batch_Size, N, Context_Length, Features]`
    *   `Batch_Size`: Number of samples processed simultaneously (e.g., 32).
    *   `N`: Number of sensors (e.g., 207).
    *   `Context_Length`: The historical look-back window (e.g., 96 time steps representing 8 hours of 5-minute data).
    *   `Features`: The variables being measured (e.g., 1 feature for traffic speed).

**The Baseline Limitation:**
Vanilla TTM expects an input shape of `[Batch_Size, Context_Length, Features]`. It flattens the `N` dimension into the batch dimension, treating every sensor as completely independent and isolated. It cannot utilize the Adjacency Matrix `A`.

---

## 2. Architecture Design: The "Pre-Adapter"

We must decide where the adapter sits relative to the TTM backbone.

1.  **Pre-Adapter (V1 Plan):** Modifies the raw data `X` using `A` *before* passing it to TTM.
2.  **Internal-Adapter:** Injects spatial attention *inside* the Transformer blocks of TTM. (Harder to implement, requires modifying HuggingFace internals).
3.  **Post-Adapter:** Modifies the output prediction *after* TTM processes the data independently.

**Decision:** We will implement the **Pre-Adapter** as the V1 baseline. It is modular, treats TTM as a black box, and is easiest to debug.

---

## 3. PyTorch Implementation Plan

Below is the conceptual PyTorch code required to execute this objective.

### Step 3.1: Load and Freeze the Backbone
We load the pre-trained TTM model and ensure its gradients are turned off. This preserves its pre-trained knowledge and drastically reduces training time/memory.

```python
import torch
import torch.nn as nn
from transformers import TTMModel

# Load pre-trained TTM from HuggingFace
ttm_backbone = TTMModel.from_pretrained("ibm/granite-ttm")

# Freeze all weights in the TTM backbone
for param in ttm_backbone.parameters():
    param.requires_grad = False
```

### Step 3.2: Write the Spatial Adapter
The adapter performs a basic Graph Convolution. It mixes data from neighboring nodes before applying a learned transformation.

```python
class SpatialAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Learnable weights for the graph convolution
        self.weight_matrix = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, X, Adjacency_Matrix):
        """
        X shape: (Batch, N, Context_Length, Features)
        A shape: (N, N)
        """
        # 1. Message Passing: Mix data across the spatial graph
        # This allows Sensor i to "see" the time-series of its neighbors
        spatial_mix = torch.matmul(Adjacency_Matrix, X) 
        
        # 2. Learnable Transformation
        enriched_X = self.weight_matrix(spatial_mix)
        
        # 3. Non-linear Activation
        output = torch.relu(enriched_X)
        
        # 4. Residual Connection (Original Data + Spatial Data)
        return X + output 
```

### Step 3.3: The Unified Urban Foundation Model
We combine the adapter and the frozen backbone into the final trainable module.

```python
class UrbanFoundationModel(nn.Module):
    def __init__(self, frozen_ttm):
        super().__init__()
        self.adapter = SpatialAdapter(feature_dim=1)
        self.ttm = frozen_ttm
        
    def forward(self, X, Adjacency_Matrix):
        Batch, N, Context_Length, Features = X.shape
        
        # 1. Apply Spatial Adapter
        # Output shape is still (Batch, N, Context_Length, Features)
        spatial_X = self.adapter(X, Adjacency_Matrix)
        
        # 2. Reshape for TTM
        # Merge Batch and N dimensions. TTM thinks it's processing a normal, 
        # independent batch of time-series, unaware that they have been spatially mixed.
        ttm_input = spatial_X.view(Batch * N, Context_Length, Features)
        
        # 3. Pass enriched data through FROZEN TTM
        prediction = self.ttm(ttm_input)
        
        # 4. Reshape back to original graph dimensions
        prediction = prediction.view(Batch, N, -1)
        
        return prediction
```

---

## 4. Training and Evaluation Strategy

**The Training Loop:**
During training, `loss.backward()` will calculate gradients for the entire computational graph. However, because `ttm_backbone.requires_grad = False`, PyTorch will only update the weights inside `self.adapter.weight_matrix`. The millions of parameters inside TTM remain untouched.

**The Validation Test:**
1.  Train the `UrbanFoundationModel` on the METR-LA dataset.
2.  Pass the same dataset through the raw `ttm_backbone` (without the adapter).
3.  Compare MAE (Mean Absolute Error).

**Success Criteria:**
If `MAE(UrbanFoundationModel) < MAE(Vanilla TTM)`, Objective 1 is complete. This proves that injecting graph structure into a frozen time-series foundation model improves accuracy for urban prediction tasks.
