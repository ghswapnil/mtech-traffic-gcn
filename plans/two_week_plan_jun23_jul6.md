# Two-Week Action Plan: June 23 → July 6, 2026

> **Rule:** Every day has a coding task. No day is reading-only.
> **Goal by July 6:** You have a working vanilla TTM baseline on METR-LA with a MAE number.

---

## Week 1: Environment + Data + First Baseline (June 23–29)

### Day 1 — Monday June 23: Environment Setup + TTM Paper (Morning)
- [ ] **Morning (2 hrs):** Read the TTM paper. You already know PatchTST and TSMixer — TTM will be fast. Focus on:
  - Section 3 (Architecture): How does adaptive patching work?
  - Section 4 (Pre-training): What datasets was it pre-trained on?
  - Table 1 (Results): What are the reported zero-shot MAE numbers?
- [ ] **Afternoon (3 hrs):** Set up your Python environment:
  ```bash
  # Create a conda environment
  conda create -n mtech python=3.10 -y
  conda activate mtech

  # Install PyTorch (with CUDA if you have a GPU)
  pip install torch torchvision

  # Install IBM granite-tsfm (TTM)
  pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.18"

  # Install other essentials
  pip install numpy pandas matplotlib scikit-learn scipy
  ```
- [ ] **Evening (1 hr):** Run the official TTM tutorial notebook from the `granite-tsfm` repo to verify your installation works. Just run it — don't modify anything.

**Deliverable:** Working Python environment + TTM tutorial runs successfully.

---

### Day 2 — Tuesday June 24: Download METR-LA + Write Dataset Class
- [ ] **Morning (3 hrs):** Download the METR-LA dataset.
  - Source: https://github.com/liyaguang/DCRNN (the original DCRNN repo has download links)
  - Files you need:
    - `metr-la.h5` — the traffic speed readings (207 sensors × 34,272 time steps)
    - `adj_mx.pkl` — the adjacency matrix (207 × 207, distance-based)
  - Inspect the data: How many sensors? What's the time resolution? What are min/max speed values?
- [ ] **Afternoon (3 hrs):** Write `src/dataset.py`:
  ```python
  # Must implement:
  # 1. Load metr-la.h5 into a numpy array (N=207, T=34272)
  # 2. Z-score normalize per sensor: (x - mean) / std
  # 3. Create sliding windows: input (T_in=512) → target (T_out=96)
  # 4. Chronological split: 70% train / 10% val / 20% test
  # 5. Return a PyTorch Dataset that yields (input, target) pairs
  ```
- [ ] **Evening (1 hr):** Verify your dataset by plotting 3 random sensors' time series using matplotlib.

**Deliverable:** `src/dataset.py` that loads METR-LA and creates train/val/test splits.

---

### Day 3 — Wednesday June 25: Vanilla TTM Zero-Shot Inference
- [ ] **All day (5-6 hrs):** Write `src/baseline_ttm.py`:
  1. Load the METR-LA test set from your dataset class.
  2. Load the pre-trained TTM model:
     ```python
     from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
     model = TinyTimeMixerForPrediction.from_pretrained(
         "ibm-granite/granite-timeseries-ttm-r2-1024-96-ft-r1"
     )
     ```
  3. Run inference on each sensor independently (channel independence).
  4. Compute MAE and RMSE on **de-normalized** predictions.
  5. Print the results.
- [ ] **Evening:** Read the DLinear paper (it's only 9 pages and very readable). This will take ~1 hour.

**Deliverable:** A printed MAE/RMSE number for vanilla TTM on METR-LA. THIS IS YOUR BASELINE.

---

### Day 4 — Thursday June 26: STGCN Paper + Load Adjacency Matrix
- [ ] **Morning (2 hrs):** Read the STGCN paper (Yu et al., IJCAI 2018). Focus on:
  - How they define the graph (distance-based thresholded Gaussian kernel)
  - The ST-Conv block (sandwich structure: temporal → spatial → temporal)
  - Table 2 (their reported MAE on METR-LA — this is another baseline number for you)
- [ ] **Afternoon (3 hrs):** Write `src/graph.py`:
  ```python
  # Must implement:
  # 1. Load adj_mx.pkl (the pre-computed adjacency matrix)
  # 2. Normalize it: D^(-1/2) * A * D^(-1/2) (symmetric normalization)
  # 3. Convert to a PyTorch sparse tensor
  # 4. Visualize: plot the adjacency matrix as a heatmap
  ```
- [ ] **Evening (1 hr):** Implement a basic 2-layer GCN module in `src/gcn.py`:
  ```python
  class GCNLayer(nn.Module):
      def forward(self, X, A_norm):
          # X: (Batch, N, Features)
          # A_norm: (N, N) normalized adjacency
          return relu(A_norm @ X @ self.W)
  ```

**Deliverable:** `src/graph.py` + `src/gcn.py` with a working GCN that you can test on dummy data.

---

### Day 5 — Friday June 27: Build the Parallel Adapter (V1)
- [ ] **All day (6 hrs):** Write `src/adapter.py` — the core of your thesis:
  ```python
  class ParallelAdapter(nn.Module):
      def __init__(self, ttm_model, gcn, fusion_mlp):
          self.ttm = ttm_model  # FROZEN
          self.gcn = gcn        # TRAINABLE
          self.fusion = fusion_mlp  # TRAINABLE
          
          # Freeze TTM
          for p in self.ttm.parameters():
              p.requires_grad = False

      def forward(self, X, A_norm):
          # Temporal branch (frozen)
          B, N, T = X.shape
          x_temp = X.reshape(B * N, 1, T)
          feat_temp = self.ttm(x_temp)  # (B*N, D_temp)
          feat_temp = feat_temp.reshape(B, N, -1)

          # Spatial branch (trainable)
          feat_spat = self.gcn(X, A_norm)  # (B, N, D_spat)

          # Fusion
          combined = torch.cat([feat_temp, feat_spat], dim=-1)
          return self.fusion(combined)  # (B, N, T_out)
  ```
- [ ] **Note:** The TTM API may not work exactly like the pseudocode above. You will need to read the `granite-tsfm` source code to understand the exact input/output shapes. Expect to spend 2-3 hours debugging this.

**Deliverable:** `src/adapter.py` with a `ParallelAdapter` class that compiles and runs on dummy data.

---

### Day 6–7 — Weekend June 28–29: Training Loop + First Adapter Result
- [ ] **Saturday (6 hrs):** Write `src/train.py`:
  1. Training loop with Adam optimizer (lr=1e-3)
  2. Loss: MSE on de-normalized predictions
  3. Validation every epoch
  4. Save best checkpoint
  5. Early stopping (patience=10)
- [ ] **Sunday (4 hrs):** Train the adapter on METR-LA.
  - Since TTM is frozen, only the GCN + MLP weights update → training should be fast (< 1 hour on a single GPU).
  - Compare: Adapter MAE vs. Vanilla TTM MAE.
- [ ] **Sunday evening (2 hrs):** Read the UniST paper (your competitor). Focus on how they handle spatial encoding.

**Deliverable:** First adapter training run complete. You now have two numbers:
| Model | MAE |
|---|---|
| Vanilla TTM (zero-shot) | ??? |
| TTM + GCN Adapter (trained) | ??? |

---

## Week 2: Experiments + Ablations + Writing (June 30 – July 6)

### Day 8 — Monday June 30: Fix Bugs + Hyperparameter Tuning
- [ ] Debug any issues from the weekend training run.
- [ ] Try different hyperparameters:
  - GCN layers: 1, 2, 3
  - Hidden dimension: 32, 64, 128
  - Learning rate: 1e-2, 1e-3, 1e-4
- [ ] Log all results in a spreadsheet or CSV file.

---

### Day 9 — Tuesday July 1: Add STGCN Baseline
- [ ] Download and run an existing STGCN implementation on METR-LA.
  - Use an existing repo like [BasicTS](https://github.com/zezhishao/BasicTS) which has STGCN, DCRNN, AGCRN all pre-implemented.
- [ ] Record STGCN's MAE/RMSE on the same test split you used.

**Deliverable:** Three-row results table:
| Model | MAE | RMSE |
|---|---|---|
| Vanilla TTM | ??? | ??? |
| STGCN (from scratch) | ??? | ??? |
| TTM + GCN Adapter | ??? | ??? |

---

### Day 10 — Wednesday July 2: Ablation Studies
- [ ] **Ablation 1:** Remove the spatial branch entirely (TTM + MLP only, no GCN). Does the GCN actually help?
- [ ] **Ablation 2:** Remove the temporal branch (GCN only, no TTM). Is the frozen TTM actually useful?
- [ ] **Ablation 3:** Try a Pre-Adapter instead of Parallel-Adapter. Which architecture wins?

---

### Day 11 — Thursday July 3: Multi-Horizon Evaluation
- [ ] Evaluate at 3 forecasting horizons: 15 min, 30 min, 60 min (3, 6, 12 steps ahead).
- [ ] Create the standard results table that every traffic paper reports.

---

### Day 12 — Friday July 4: Cross-Dataset Transfer (PEMS-BAY)
- [ ] Download the PEMS-BAY dataset (325 sensors, Bay Area).
- [ ] Test your METR-LA-trained adapter on PEMS-BAY **without retraining** (zero-shot transfer).
- [ ] Record the transfer MAE. Even if it's bad, this is valuable data.

---

### Day 13–14 — Weekend July 5–6: Document + Reflect + Plan Next Phase
- [ ] Write a 2-page internal report summarizing:
  - What worked, what didn't
  - Best adapter configuration
  - All baseline comparisons
  - Key observations and surprises
- [ ] Plan the next phase: spectral wavelet adapter (Direction 1 from our math discussion)
- [ ] Update your `src/` directory README with clear instructions

---

## End-of-Sprint Success Criteria

By **July 6**, you should have:

| Checkpoint | Status |
|---|---|
| Working Python environment | ☐ |
| METR-LA loaded and visualized | ☐ |
| Vanilla TTM baseline MAE computed | ☐ |
| GCN adapter implemented and trained | ☐ |
| Adapter vs. TTM vs. STGCN comparison table | ☐ |
| Ablation studies completed | ☐ |
| Multi-horizon evaluation (15/30/60 min) | ☐ |
| Zero-shot transfer to PEMS-BAY attempted | ☐ |
| 2-page internal report written | ☐ |

> [!CAUTION]
> **If you reach July 6 and your `src/` directory is still empty, you are in serious trouble.**
> The reading phase is OVER. You know enough theory to build.
> Every day you delay coding is a day closer to the deadline with nothing to show.
