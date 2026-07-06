# Revised Two-Week Plan: June 7 – June 21, 2026 (Aggressive)

> **Goal by June 21:** Vanilla TTM baseline on METR-LA + your first spatial adapter prototype running (even if results are bad).

---

## Week 1: June 7–13

### Day 1 — Saturday June 7: Data + GCN paper
**Morning (3 hrs):**
- [ ] Download METR-LA dataset
- [ ] Write a Python script: load .h5, print shapes, plot 3 sensors over 1 week
- [ ] Load adjacency matrix, visualize the sensor graph with networkx

**Afternoon (3 hrs):**
- [ ] Read GCN paper (Kipf & Welling) — Sections 1-3 only
- [ ] You already know the math from Ch 5. Focus on: self-loops, renormalization trick, number of layers
- [ ] This should take 2 hours, not a full day

**Evening (1 hr):**
- [ ] Read GraphSAGE paper — abstract + Section 3 (the sampling algorithm) only

### Day 2 — Sunday June 8: Code a GCN from scratch
**Full day (6 hrs):**
- [ ] Code a 2-layer GCN on Cora dataset from scratch in PyTorch
  - NOT using PyG's GCNConv layer — write the matrix multiplication yourself
  - `H' = σ(Ã @ H @ W)` — that's literally 3 lines of math
- [ ] Train it, get ~80% accuracy on node classification
- [ ] Save as `src/experiments/toy_gcn_from_scratch.py`
- [ ] **Checkpoint: you can now say "I implemented a GCN"**

### Day 3 — Monday June 9: TTM paper (1st read) + clone repo
**Morning (3 hrs):**
- [ ] Read TTM paper — first pass, big picture only
- [ ] Key questions: input format? output format? how big? MLP-Mixer not Transformer.

**Afternoon (3 hrs):**
- [ ] Clone `ibm-granite/granite-tsfm`
- [ ] Set up environment, install dependencies
- [ ] Run their hello-world tutorial — make TTM produce ANY prediction
- [ ] Save as `src/experiments/ttm_hello_world.py`

### Day 4 — Tuesday June 10: METR-LA DataLoader
**Full day (6 hrs):**
- [ ] Write proper PyTorch Dataset for METR-LA:
  - Chronological split: 70% train / 10% val / 20% test
  - Sliding window: 12 input steps → 12 prediction steps (5-min intervals)
  - Z-score normalization per sensor
- [ ] Write a DataLoader that batches it
- [ ] Test: load one batch, print shapes, verify it looks right
- [ ] Save as `src/data/metrla_dataset.py`

### Day 5 — Wednesday June 11: Vanilla TTM on METR-LA
**Full day (6 hrs):**
- [ ] Feed METR-LA into TTM (each sensor independently)
- [ ] Compute MAE, RMSE, MAPE on test set
- [ ] Report for 15-min (3 steps), 30-min (6 steps), 60-min (12 steps)
- [ ] **Write down THE NUMBERS. This is your baseline.**
- [ ] Save as `src/experiments/ttm_metrla_baseline.py`

### Day 6 — Thursday June 12: Read STGCN + TTM paper (2nd read)
**Morning (3 hrs):**
- [ ] Read STGCN paper — focus on architecture figure and Section 3
- [ ] Now that you've coded a GCN and run TTM, STGCN = "GCN + temporal conv"
- [ ] Note the sandwich structure: temporal → spatial → temporal

**Afternoon (3 hrs):**
- [ ] Re-read TTM paper — focus on:
  - How does patching work exactly?
  - How does channel mixing work?
  - What does fine-tuning look like vs zero-shot?
- [ ] Read Ch 12.3 (Traffic Prediction) from DLG book — 1 hour, it's short

### Day 7 — Friday June 13: Design adapter + start building
**Morning (2 hrs):**
- [ ] Design your spatial adapter V1 on paper
- [ ] Decide: Pre-Adapter, Post-Adapter, or Parallel? (try Pre first)
- [ ] Write architecture doc as `src/docs/adapter_design_v1.md`

**Afternoon (4 hrs):**
- [ ] Start implementing SpatialAdapter class in PyTorch
- [ ] Wire it up: METR-LA → SpatialAdapter → TTM
- [ ] Don't worry if it doesn't train yet — just make the forward pass work
- [ ] Save as `src/models/spatial_adapter.py`

---

## Week 2: June 14–21

### Day 8 — Saturday June 14: Get adapter training
- [ ] Write the training loop for UrbanFoundationModel
- [ ] Freeze TTM weights, only train adapter
- [ ] Run 1 epoch — does the loss go down? If yes, you're golden. If no, debug.
- [ ] Save as `src/train.py`

### Day 9 — Sunday June 15: First adapter results
- [ ] Train for 20-50 epochs
- [ ] Compute MAE on test set
- [ ] Compare: Vanilla TTM MAE vs. TTM + Adapter MAE
- [ ] **Even if the adapter makes things WORSE, that's useful information**
- [ ] Save results in `src/results/experiment_01.md`

### Day 10 — Monday June 16: Debug and iterate
- [ ] If adapter helps → great, try adding a 2nd GCN layer, or residual connections
- [ ] If adapter hurts → try Post-Adapter (GCN after TTM, not before)
- [ ] Run at least 2 architecture variants and record results
- [ ] This is where real research happens

### Day 11 — Tuesday June 17: Run STGCN baseline
- [ ] Install and run an existing STGCN implementation on METR-LA
- [ ] Use a public repo (e.g., from PyG or LibCity)
- [ ] Record MAE/RMSE/MAPE — this is your key comparison baseline
- [ ] Now you have 3 numbers: Vanilla TTM vs. TTM+Adapter vs. STGCN

### Day 12 — Wednesday June 18: Read remaining papers (speed round)
Now that you've BUILT things, read these fast — they'll make much more sense:
- [ ] **PatchTST** — 2 hours (you now understand what TTM does, this is its ancestor)
- [ ] **DCRNN** — 1.5 hours (you now understand GCNs + traffic)
- [ ] **TSMixer** — 1 hour (TTM's direct parent)
- [ ] Take notes on any ideas you want to try in your adapter

### Day 13 — Thursday June 19: Improve adapter based on paper insights
- [ ] Did PatchTST give you ideas about patching in your adapter?
- [ ] Did DCRNN give you ideas about using diffusion instead of GCN?
- [ ] Implement ONE improvement and test it
- [ ] Record results

### Day 14 — Friday June 20-21: Polish + plan next phase
- [ ] Write a results summary: what worked, what didn't, what to try next
- [ ] Update tracker.html
- [ ] Plan Week 3-4 (deeper adapter experiments, more datasets)
- [ ] Talk to your advisor about early results

---

## What you'll have by June 21

```
✅ METR-LA data pipeline (reusable for all future experiments)
✅ Toy GCN coded from scratch (proof you understand GNNs)
✅ Vanilla TTM baseline numbers on traffic data
✅ First spatial adapter prototype — trained and evaluated
✅ STGCN baseline for comparison
✅ 3-way comparison: TTM vs. TTM+Adapter vs. STGCN
✅ 6 papers read WITH context (GCN, GraphSAGE, TTM, STGCN, PatchTST, DCRNN)
✅ Architecture decisions documented
```

---

## The Rule

**Every paper you read in these two weeks must be AFTER you've coded something related to it.** Not before.

```
❌ Read PatchTST → Read DCRNN → Read TSMixer → eventually code something
✅ Code GCN → read GCN paper → run TTM → read TTM paper → build adapter → read STGCN/PatchTST
```

Papers click 10x faster after you've wrestled with the code.
