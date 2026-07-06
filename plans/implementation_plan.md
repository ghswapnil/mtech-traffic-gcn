# METR-LA Traffic Forecasting: TTM Baseline & GCN Adapter

This plan details the design and coding steps for evaluating the pre-trained IBM TTM model on the METR-LA dataset (temporal baseline) and building the spatial Graph Neural Network (GCN) adapter to fuse spatial-temporal information.

## User Review Required

> [!IMPORTANT]
> Since we are working with a frozen pre-trained time-series foundation model, we must adapt the data shapes from our 2D spatial-temporal grid `(batch_size, context_len, num_sensors)` to the 3D tensor shape expected by TTM `(batch_size, context_len, num_features)`. We will treat the 207 sensors as 207 parallel feature channels. TTM handles this internally in a channel-independent manner.

---

## Proposed Changes

### 1. Temporal Baseline Evaluation

#### [NEW] [baseline_ttm.py](file:///Users/swapnilaggarwal/M.Tech%20Project/src/baseline_ttm.py)
A script to run zero-shot inference using the pre-trained IBM TTM model on the METR-LA test split.
*   **Data Loading:** Uses the verified `load_metr_la_data()` function to load test windows.
*   **Model Loading:** Instantiates the pre-trained `ibm-granite/granite-timeseries-ttm-r2` model using `TinyTimeMixerForPrediction`.
*   **Inference Loop:** 
    *   Iterates through test batches.
    *   Feeds standardized context windows `(batch, 512, 207)` into the TTM model.
    *   Generates predictions of shape `(batch, 96, 207)`.
    *   Inverts the normalization using `scaler.inverse_transform()`.
*   **Metrics Computation:** Computes **MAE**, **RMSE**, and **MAPE** for the standard traffic forecasting horizons:
    *   **15 minutes** (step index 2)
    *   **30 minutes** (step index 5)
    *   **60 minutes** (step index 11)

---

### 2. Graph Normalization Utility

#### [NEW] [graph.py](file:///Users/swapnilaggarwal/M.Tech%20Project/src/data/graph.py)
A module to load and process the road network graph adjacency matrix (`adj_mx.pkl`).
*   **Self-loops:** Adds self-loops to the adjacency matrix ($\tilde{A} = A + I_N$).
*   **Symmetric Normalization:** Calculates the degree matrix $\tilde{D}$ and returns the normalized laplacian representation:
    $$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
*   **PyG Format:** Formats the adjacency matrix into PyG-compatible `edge_index` and `edge_weight` tensors.

---

### 3. Model Architecture

#### [NEW] [gcn.py](file:///Users/swapnilaggarwal/M.Tech%20Project/src/models/gcn.py)
A PyTorch GCN module to process spatial interactions between the road sensors.
*   Uses `GCNConv` layers from `torch_geometric.nn`.
*   Computes spatial sensor representations at each time step.

#### [NEW] [adapter.py](file:///Users/swapnilaggarwal/M.Tech%20Project/src/models/adapter.py)
The core adapter wrapper.
*   Keeps `TinyTimeMixerForPrediction` frozen.
*   Sends input context to the GCN branch and TTM branch.
*   Combines the representations using a learnable Multi-Layer Perceptron (MLP) forecasting head.

---

### 4. Training Pipeline

#### [NEW] [train.py](file:///Users/swapnilaggarwal/M.Tech%20Project/src/train.py)
A training script to optimize the adapter.
*   Sets up training loop using Adam optimizer.
*   Ensures TTM parameters remain frozen (`requires_grad = False`).
*   Saves the best model checkpoints based on validation loss.

---

## Verification Plan

### Automated Tests
*   **Run Baseline:** Evaluate the zero-shot model and print metrics:
    ```bash
    "./venv/bin/python" src/baseline_ttm.py
    ```
*   **Run Training:** Run the adapter training script and ensure loss decreases:
    ```bash
    "./venv/bin/python" src/train.py
    ```
