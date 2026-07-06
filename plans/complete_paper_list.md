# Complete Audited Paper List for M.Tech Project

> **Exhaustive list** of ALL papers relevant to the project.
> Sources: Book chapters 2, 5, 8, 12 + project roadmap + evaluation checklist + missing paradigms assessment.
> Nothing is excluded — even papers covered by the book are listed.
>
> Last audited: June 9, 2026

---

## Reading Depth Guide

| Depth | What you do | Time |
|---|---|---|
| **Deep** | Read fully, take notes, understand the math, look at code | 1-3 days |
| **Standard** | Read abstract, intro, method, results. Skip proofs. | 3-6 hours |
| **Targeted** | Read only specific sections (noted below) | 1-2 hours |
| **Skim** | Abstract + figures + results tables only | 30-60 min |

---

## PART A: Papers from the Book (Chapters 2, 5, 8, 12)

### From Chapter 2: Foundations of Graphs

| # | Paper | Year | Venue | Depth | Relevance to your project |
|---|---|---|---|---|---|
| A1 | **The Emerging Field of Signal Processing on Graphs** (Shuman et al.) | 2013 | IEEE SPM | Targeted: Sections 1-4 | Graph Fourier Transform, Laplacian — the spectral foundation behind GCN. Read Sections 1-4 for intuition. |
| A2 | **Spectral Graph Theory** (Chung & Graham) | 1997 | Book | Skip | Reference textbook. Ch 2 of DLG book already covers what you need. |
| A3 | **Network Analysis in the Social Sciences** (Borgatti et al.) | 2009 | Science | Skip | General graph science. Not technically relevant. |
| A4 | **Friends and Neighbors on the Web** (Adamic & Adar) | 2003 | Social Networks | Skip | Link prediction heuristic. Cited in Ch 2 for context. |
| A5 | **Abstract Meaning Representation for Sembanking** (Banarescu et al.) | 2013 | LAW | Skip | NLP graph example. Not relevant. |
| A6 | **Network Datasets** (Leskovec & Krevl / Rossi & Ahmed) | 2014/2015 | SNAP/KONECT | Skip | Dataset repositories. Not papers to read. |

---

### From Chapter 5: Graph Neural Networks

#### The Spectral → GCN Derivation Chain (MUST READ)

| # | Paper | Year | Venue | Pages | Depth | What to read |
|---|---|---|---|---|---|---|
| B1 | **The Graph Neural Network Model** (Scarselli et al.) | 2005/2008 | IEEE TNN | 20 | Targeted | Sections 1-3 only. The ORIGINAL GNN — iterative message passing until convergence. Historical foundation. |
| B2 | **Wavelets on Graphs via Spectral Graph Theory** (Hammond et al.) | 2011 | ACHA | 37 | Targeted | **Pages 16-22 ONLY.** Chebyshev polynomial approximation for spectral filters (Eq. 8-12). |
| B3 | **Spectral Networks and Locally Connected Networks on Graphs** (Bruna et al.) | 2013 | arXiv | 14 | Targeted | Sections 1-3. First to apply spectral convolution via neural networks on graphs. |
| B4 | **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebNet)** (Defferrard et al.) | 2016 | NeurIPS | 10 | Standard | Sections 1-3 fully. Eq. 3-5. The bridge: uses Chebyshev polynomials as learnable graph filters. |
| B5 | **Semi-Supervised Classification with Graph Convolutional Networks (GCN)** (Kipf & Welling) | 2017 | ICLR | 14 | ⭐ **Deep** | **Read fully.** Your adapter IS this. Simplifies ChebNet to first-order: `H' = σ(ÃHW)`. |

#### Spatial-based Graph Filters

| # | Paper | Year | Venue | Pages | Depth | What to read |
|---|---|---|---|---|---|---|
| B6 | **Gated Graph Sequence Neural Networks (GG-S-NN)** (Li et al.) | 2015 | arXiv/ICLR | 16 | Targeted | Sections 1-3. GRU-based message passing. Alternative to GCN — recurrent instead of convolutional. |
| B7 | **Neural Message Passing for Quantum Chemistry (MPNN)** (Gilmer et al.) | 2017 | ICML | 14 | Standard | Sections 1-3. Skip quantum chemistry parts. Unified message → aggregate → update framework. |
| B8 | **Inductive Representation Learning on Large Graphs (GraphSAGE)** (Hamilton et al.) | 2017 | NeurIPS | 11 | Standard | Read fully. Handles **unseen nodes** via neighbor sampling. Critical for cross-city transfer (Obj 5). |
| B9 | **Graph Attention Networks (GAT)** (Veličković et al.) | 2018 | ICLR | 12 | Standard | Read fully. Attention-weighted neighbor aggregation. You may use GAT instead of GCN in your adapter. |
| B10 | **Diffusion-Convolutional Neural Networks (DCNN)** (Atwood & Towsley) | 2016 | NeurIPS | 9 | Skim | Models diffusion on graphs. DCRNN (your baseline) builds on this idea. |
| B11 | **Geometric Matrix Completion with Multi-Graph Neural Networks (MoNet)** (Monti et al.) | 2017 | NeurIPS | 11 | Skim | Mixture model spatial convolution. Alternative filter design. |
| B12 | **Dynamic Edge-Conditioned Filters (ECC)** (Simonovsky & Komodakis) | 2017 | CVPR | 10 | Skim | Edge-specific filter weights. Relevant concept if your graph has different edge types. |
| B13 | **Learning Convolutional Neural Networks for Graphs (PATCHY-SAN)** (Niepert et al.) | 2016 | ICML | 10 | Skim | Converts graph patches to sequences. Alternative paradigm you won't use, but good to cite. |

#### Graph Pooling

| # | Paper | Year | Venue | Pages | Depth | What to read |
|---|---|---|---|---|---|---|
| B14 | **Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)** (Ying et al.) | 2018 | NeurIPS | 11 | Skim | Learned hierarchical pooling. Not directly needed (you do node-level, not graph-level tasks). |
| B15 | **Self-Attention Graph Pooling (SAGPool)** (Lee et al.) | 2019 | ICML | 10 | Skim | Attention-based node selection for pooling. Same — graph classification, not your task. |
| B16 | **Graph U-Nets** (Gao & Ji) | 2019 | ICML | 10 | Skim | Pool/unpool operations. Not directly needed. |
| B17 | **Large-Scale Learnable Graph Convolutional Networks** (Gao et al.) | 2018 | KDD | 9 | Skim | Learnable pooling. Not directly needed. |
| B18 | **Kronecker Attention Networks** (Gao et al.) | 2020 | KDD | 9 | Skim | Advanced attention for graphs. Cited in Ch 5 Further Reading. |

#### GNN Expressiveness & Theory

| # | Paper | Year | Venue | Depth | What to read |
|---|---|---|---|---|---|
| B19 | **Weisfeiler and Leman Go Neural (WL-GNN)** (Morris et al.) | 2019 | AAAI | Skim | GNN expressiveness linked to WL isomorphism test. Theoretical — good to cite. |
| B20 | **Deep Graph Kernels** (Yanardag & Vishwanathan) | 2015 | KDD | Skim | Graph kernels. Cited in Further Reading. Historical. |
| B21 | **GeniePath: Adaptive Receptive Paths** (Liu et al. / Yuan & Ji) | 2019 | AAAI | Skim | Adaptive depth. Cited in Further Reading. |
| B22 | **Deep Graph Infomax** (Veličković et al.) | 2019 | ICLR | Skim | Self-supervised learning on graphs. Could be relevant for adapter pre-training idea. |

#### Libraries & Surveys

| # | Paper | Year | Venue | Depth | What to read |
|---|---|---|---|---|---|
| B23 | **Fast Graph Representation Learning with PyTorch Geometric** (Fey & Lenssen) | 2019 | ICLR-W | Skim | The PyG library you'll use. Read to understand the API. |
| B24 | **A Comprehensive Survey on Graph Neural Networks** (Wu et al.) | 2020 | IEEE TNNLS | Targeted | Read Section 3 (taxonomy) when writing your related work. |
| B25 | **Graph Neural Networks: A Review** (Zhou et al.) | 2018 | AI Open | Skim | Alternative survey. Cite in thesis. |
| B26 | **Deep Learning on Graphs: A Survey** (Zhang et al.) | 2018 | IEEE TKDE | Skim | Alternative survey. Cite in thesis. |

---

### From Chapter 8: GNNs on Complex Graphs

| # | Paper | Year | Venue | Depth | What to read |
|---|---|---|---|---|---|
| C1 | **Heterogeneous Graph Neural Network (HetGNN)** (Zhang et al.) | 2019 | KDD | Standard | Read fully when starting Objective 3. How to handle different node/edge types. |
| C2 | **Heterogeneous Graph Attention Network (HAN)** (Wang et al.) | 2019 | WWW | Standard | Hierarchical attention for heterogeneous graphs. Directly relevant to your multi-domain urban graph. |
| C3 | **EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs** (Pareja et al.) | 2019 | AAAI | Standard | GCN weights evolve over time. Relevant to your time-varying traffic graph. |
| C4 | **DySAT: Dynamic Self-Attention Network** (Sankar et al.) | 2018 | arXiv | Targeted | Self-attention for dynamic graphs. Read Sections 1-3 when working on temporal graphs. |
| C5 | **Signed Graph Convolutional Network** (Derr et al.) | 2018 | ICDM | Skim | Signed graphs (positive/negative edges). Not directly relevant. |
| C6 | **Hypergraph Neural Networks** (Feng et al.) | 2019 | AAAI | Skim | Hypergraph convolution. Not directly relevant unless you model group relationships. |
| C7 | **Heterogeneous Hyper-Network Embedding** (Baytas et al.) | 2018 | ICDM | Skim | Heterogeneous + hypergraph. Niche. |
| C8 | **Variational Graph Auto-Encoder** (Kipf & Welling) | 2016 | arXiv | Skim | VAE on graphs. Referenced in Ch 8. Not directly used. |

---

### From Chapter 12: GNNs in Data Mining (Section 12.3)

| # | Paper | Year | Venue | Depth | What to read |
|---|---|---|---|---|---|
| D1 | **Spatio-Temporal Graph Convolutional Networks (STGCN)** (Yu et al.) | 2018 | IJCAI | ⭐ **Deep** | Already in your roadmap. THE foundational ST-GNN for traffic. |
| D2 | **Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)** (Guo et al.) | 2019 | AAAI | Standard | Adds attention to spatial AND temporal dimensions. Stronger baseline than basic STGCN. |
| D3 | **Traffic Transformer** (Wang et al.) | 2020 | — | Standard | Transformer for traffic with GNN spatial module. Referenced in Ch 12.3. |
| D4 | **A Hybrid Model for Spatiotemporal Forecasting of PM2.5** (Qi et al.) | 2019 | KDD | Skim | Air quality forecasting with GNN. Similar ST framework, different domain. Good for cross-domain ideas. |

---

## PART B: Papers from Project Roadmap (Not in the Book)

### Transformer & Language Model Foundations

| # | Paper | Year | Venue | Depth | Why |
|---|---|---|---|---|---|
| E1 | **Attention Is All You Need** (Vaswani et al.) | 2017 | NeurIPS | Targeted: Section 3 | The Transformer architecture. Also referenced in Ch 5. |
| E2 | **BERT** (Devlin et al.) | 2019 | NAACL | Targeted: Sections 1-3 | Masked pre-training paradigm. TTM uses similar approach. |
| E3 | **word2vec Explained** (Goldberg & Levy) | 2014 | arXiv | Skim | Embeddings concept. Background knowledge. |
| E4 | **Efficient Estimation of Word Representations** (Mikolov et al.) | 2013 | arXiv | Skim | Original Word2Vec. Background. |

### Time Series Deep Learning

| # | Paper | Year | Venue | Depth | Why |
|---|---|---|---|---|---|
| E5 | **PatchTST** (Nie et al.) | 2023 | ICLR | ⭐ **Deep** | Patching concept — the foundation TTM is built on. |
| E6 | **DLinear** (Zeng et al.) | 2023 | AAAI | Skim | "Are Transformers needed?" Counterpoint. Good to cite. |
| E7 | **Informer** (Zhou et al.) | 2021 | AAAI | Targeted: Section 3 | ProbSparse attention. Historical context for TS Transformers. |
| E8 | **Autoformer** (Wu et al.) | 2021 | NeurIPS | Targeted: Section 3 | Series decomposition. Potentially useful idea. |
| E9 | **TSMixer** (Chen et al.) | 2023 | KDD | ⭐ **Deep** | TTM's direct parent. MLP-Mixer for time series. |
| E10 | **TS2Vec** (Yue et al.) | 2022 | AAAI | Skim | Contrastive learning for TS. Alternative paradigm. |

### Time Series Foundation Models

| # | Paper | Year | Venue | Depth | Why |
|---|---|---|---|---|---|
| E11 | **TTM: Tiny Time Mixers** (Ekambaram et al., IBM) | 2024 | NeurIPS | ⭐⭐ **Deep (3x)** | YOUR BACKBONE. Read three times. |
| E12 | **Chronos** (Ansari et al., Amazon) | 2024 | ICML | Standard | Competitor — tokenization approach. |
| E13 | **TimesFM** (Das et al., Google) | 2024 | ICML | Standard | Competitor — decoder-only approach. |
| E14 | **Moirai** (Woo et al., Salesforce) | 2024 | ICML | Targeted: Sections 1-3 | Any-variate attention. |
| E15 | **Time-LLM** (Jin et al.) | 2024 | ICLR | Targeted: Sections 1-3 | Reprogramming paradigm. |
| E16 | **Lag-Llama** (Rasul et al.) | 2023 | NeurIPS | Skim | Probabilistic forecasting. |
| E17 | **One Fits All** (Zhou et al.) | 2023 | NeurIPS | Skim | Cross-modal adaptation. |

### Spatio-Temporal Models (Beyond the Book)

| # | Paper | Year | Venue | Depth | Why |
|---|---|---|---|---|---|
| E18 | **DCRNN** (Li et al.) | 2018 | ICLR | ⭐ **Deep** | Diffusion convolution for traffic. Key baseline. |
| E19 | **Graph WaveNet** (Wu et al.) | 2019 | IJCAI | Standard | Learns graph from data. Adaptive adjacency. |
| E20 | **AGCRN** (Bai et al.) | 2020 | NeurIPS | Standard | Adaptive graph + recurrent. Baseline in your eval checklist. |
| E21 | **GPT-ST** | 2023 | NeurIPS | Targeted: Sections 1-3 | Pre-training for ST graphs. |
| E22 | **UniST** (Yuan et al.) | 2024 | KDD | ⭐ **Deep** | Your closest competitor. |
| E23 | **UrbanGPT** (Li et al.) | 2024 | KDD | Standard | LLM reprogramming for ST data. |
| E24 | **CISTGNN** | 2024 | — | ⭐ **Deep** | Causal STGNNs. Core for Objective 6. |
| E25 | **Cross-City Transfer Learning** | 2018 | IJCAI | Standard | Benchmark for Objective 5. |

### Urban Domain

| # | Paper | Year | Venue | Depth | Why |
|---|---|---|---|---|---|
| E26 | **UrbanCLIP** | 2024 | WWW | Targeted | Multimodal urban understanding. |
| E27 | **BIGCity** | 2025 | ICDE | Standard | Latest unified urban model. |
| E28 | **TEMPO** | 2024 | ICLR | Skim | Prompt-based adaptation. |
| E29 | **STD-PLM** | 2025 | AAAI | Skim | Adapting PLMs for ST data. |

---

## PART C: Read When Needed (Not Scheduled)

| Paper | When | Why |
|---|---|---|
| **Urban Foundation Models Survey** (KDD 2024) | Thesis writing | Read relevant sections for Chapter 2 |
| **Physics-Informed Neural Networks** (Raissi et al., 2019) | If adding physics constraints | Method section only |
| **PCMCI** (Runge et al.) | Objective 6, Month 10 | Causal discovery algorithm |
| **The Book of Why** (Judea Pearl) | Objective 6, Month 10 | do-calculus intuition |
| **UTD19 Dataset Paper** | Objective 5, Month 8 | 40-city dataset structure |

---

## Summary

| Category | Count |
|---|---|
| Book Ch 2 papers | 6 (1 targeted, 5 skip) |
| Book Ch 5 papers | 26 (5 deep/standard, 8 targeted, 13 skim) |
| Book Ch 8 papers | 8 (2 standard, 2 targeted, 4 skim) |
| Book Ch 12 papers | 4 (1 deep, 1 standard, 2 skim) |
| Roadmap papers (not in book) | 29 |
| Read-when-needed | 5 |
| **TOTAL** | **78 items** |

### Papers you read DEEPLY: 13
B5 (GCN), E5 (PatchTST), E9 (TSMixer), E11 (TTM ×3), D1/E18 (STGCN, DCRNN), E22 (UniST), E24 (CISTGNN), B7 (MPNN), B8 (GraphSAGE), B9 (GAT)

### Papers you read at STANDARD depth: ~15
ChebNet, Chronos, TimesFM, Graph WaveNet, AGCRN, UrbanGPT, BIGCity, Cross-City Transfer, HetGNN, HAN, EvolveGCN, ASTGCN, BERT, etc.

### Papers you SKIM: ~30
Everything else — abstract + figures + results tables.

### Papers you SKIP entirely: ~15
Ch 2 general graph theory references, NLP/CV/Healthcare papers from Ch 10-11-13, social network papers from Ch 12.2.

> [!IMPORTANT]
> **You do NOT read all 78 papers before coding.** Read the 13 deep-read papers interleaved with coding over Months 1-3. Read standard/skim papers as you encounter them during implementation. Read Ch 8 papers when starting Objective 3 (Month 4). Read CISTGNN when starting Objective 6 (Month 10).
