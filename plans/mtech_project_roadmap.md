# M.Tech Project Roadmap: Foundation Models for Urban Mobility

> **Project Title:** Building Foundation Models for Urban Mobility and Smart Cities
> **Core Idea:** Adapt IBM Granite Time-Series (TTM) as backbone for urban tasks (congestion prediction, route planning, incident detection, energy-aware traffic management)
> **Timeline:** 12 months total | **Intensive Phase:** May 2026 – August 2026 (3 months free)
> **Goal:** World-class project → PhD admission at top universities

---

## 1. Your Thesis in One Sentence

**"One causal spatial adapter turns a pre-trained time-series model into a unified urban prediction system — handling traffic, energy, and transit across cities while learning true causal flow rather than spurious correlations."**

| Claim | What You Show | Why It's Novel |
|---|---|---|
| **Causal spatial adapters for TS-FMs** | Causal graph-structure adapters injected into frozen TTM outperform vanilla TTM and match/beat purpose-built ST-GNNs | Nobody has designed causal adapters for TS foundation models |
| **Multi-domain heterogeneous urban graph** | One model ingests traffic + energy + transit data via a heterogeneous graph, and cross-domain signals improve each prediction | No existing TS-FM handles cross-domain urban data |
| **Cross-city generalization** | The system transfers across cities with minimal fine-tuning, tested on 40+ cities | Foundation model + spatial adapter + transfer at this scale is new |

---

## 2. Math Prerequisites (Do Daily, Ongoing)

> [!CAUTION]
> **Build this foundation alongside your paper reading.** Without this, the GCN paper will be incomprehensible.

### Linear Algebra
- Vectors and vector spaces
- Matrix multiplication and its geometric meaning
- Linear transformations — what a matrix "does" to a vector
- Basis, span, and linear independence
- Eigenvalues and eigenvectors — what they mean geometrically
- Symmetric matrices and their properties (real eigenvalues, orthogonal eigenvectors)
- Matrix decomposition — eigendecomposition intuition

### Graph Theory & Spectral Graph Theory
- Graphs: nodes, edges, adjacency matrix, degree matrix
- Graph Laplacian: L = D - A (definition, properties, why it matters)
- Properties of L: positive semi-definite, smallest eigenvalue is 0
- Number of zero eigenvalues = number of connected components
- Fiedler value (2nd smallest eigenvalue) = algebraic connectivity
- Fiedler vector = how to partition a graph into two clusters
- Graph Fourier Transform (GFT): eigenvectors of L form a basis for signals on graphs
- Spectral filtering on graphs: multiplying in the frequency domain

### Signal Processing & Convolution
- What is convolution (1D): sliding a filter over a signal
- What is convolution (2D): how image filters work (blur, edge detection)
- Frequency domain intuition: convolution in time = multiplication in frequency
- Why convolution on graphs requires spectral methods (no "sliding" on irregular structures)
- Connection: spectral graph convolution → ChebNet approximation → GCN simplification

> [!TIP]
> **The key insight:** You do NOT need to deeply understand all spectral math. Kipf & Welling's whole point was to **simplify it away** into: *"Average each node's features with its neighbors' features, then multiply by learnable weights."* The spectral derivation is background — the final formula is simple matrix multiplication.

---

## 3. Reading List (~30 Papers)

### Phase A: Foundations

| # | Paper | Why | Time |
|---|---|---|---|
| 1 | **word2vec Explained** (Goldberg & Levy, 2014) | Mathematical derivation of Word2Vec | 0.5 day |
| 2 | **Efficient Estimation of Word Representations in Vector Space** (Mikolov et al., 2013) | The original Word2Vec — embeddings concept | 0.5 day |
| 3 | **Attention Is All You Need** (NeurIPS'17) | The Transformer. Non-negotiable. | 1.5 days |
| 4 | **BERT** (NAACL'19) | Masked pre-training paradigm that TTM uses | 0.5 day |

### Phase A.5: GNN Foundations

| # | Paper | Why | Time |
|---|---|---|---|
| 4 | **GCN** (ICLR'17) — Kipf & Welling | **THE** foundational GNN paper. Cannot understand STGCN without this. | 1.5 days |
| 5 | **MPNN** (ICML'17) — Gilmer et al. | Unified message-passing framework: message → aggregate → update. | 1 day |
| 6 | **GAT** (ICLR'18) — Veličković et al. | Attention on graphs — lets nodes dynamically weight neighbors. | 1 day |

### Phase B: Time Series Deep Learning

| # | Paper | Why | Time |
|---|---|---|---|
| 7 | **PatchTST** (ICLR'23) | **CRITICAL** — Patching concept, basis for TTM | 1.5 days |
| 8 | **DLinear** (AAAI'23) | Counterpoint — simple linear models can compete | 0.5 day |
| 9 | **Informer** (AAAI'21) | First major Transformer for time series | 1 day |
| 10 | **Autoformer** (NeurIPS'21) | Series decomposition + auto-correlation | 1 day |
| 11 | **TSMixer** (KDD'23) | **CRITICAL** — Direct ancestor of TTM | 1 day |
| 12 | **TS2Vec** (AAAI'22) | **SKIM ONLY** — alternate paradigm | 0.5 day |

### Phase C: Time Series Foundation Models

| # | Paper | Why | Time |
|---|---|---|---|
| 13 | **TTM: Tiny Time Mixers** (NeurIPS'24) — IBM | **YOUR BACKBONE. Read 3 times.** | 3 days |
| 14 | **Chronos** (ICML'24) — Amazon | Key competitor — tokenization approach | 1 day |
| 15 | **TimesFM** (ICML'24) — Google | Key competitor — decoder-only approach | 1 day |
| 16 | **Moirai** (ICML'24) — Salesforce | Any-variate attention | 1 day |
| 17 | **Time-LLM** (ICLR'24) | Reprogramming paradigm | 1 day |
| 18 | **Lag-Llama** (NeurIPS'23) | **SKIM ONLY** | 0.5 day |
| 19 | **One Fits All** (NeurIPS'23) | **SKIM ONLY** | 0.5 day |

### Phase D: Spatio-Temporal Models

| # | Paper | Why | Time |
|---|---|---|---|
| 20 | **STGCN** (IJCAI'18) | Foundational ST-GNN — GCN + temporal conv | 1 day |
| 21 | **DCRNN** (ICLR'18) | Diffusion on graphs for traffic | 1 day |
| 22 | **Graph WaveNet** (IJCAI'19) | Adaptive adjacency — learns graph from data | 0.5 day |
| 23 | **GPT-ST** (NeurIPS'23) | Pre-training for ST graphs | 1.5 days |
| 24 | **UniST** (KDD'24) | **CRITICAL** — Closest to what you're building | 2 days |
| 25 | **UrbanGPT** (KDD'24) | LLM reprogramming for ST data | 1.5 days |
| 26 | **CISTGNN** (2024) | **CRITICAL** — Causal STGNNs and confounder mitigation | 1.5 days |
| 26b| **Cross-City Transfer** (IJCAI'18) | Transfer baseline for urban settings | 0.5 day |

### Phase E: Urban Domain

| # | Paper | Why | Time |
|---|---|---|---|
| 27 | **UrbanCLIP** (WWW'24) | Multimodal urban understanding | 1 day |
| 28 | **BIGCity** (ICDE'25) | Latest unified urban model | 1.5 days |
| 29 | **TEMPO** (ICLR'24) | Prompt-based adaptation | 0.5 day |
| 30 | **STD-PLM** (AAAI'25) | Adapting PLMs for ST data | 1 day |

**Also read (when needed, not scheduled):**
- Urban Foundation Models Survey (KDD'24, arXiv 2402.01749) — read sections as needed, not cover-to-cover
- Raissi et al., "Physics-Informed Neural Networks" (2019) — METHOD section only, in Month 4

---

## 4. CS224W Videos to Watch (19 out of 60)

**With GNN papers (Phase A.5):**

| Video # | Title | Why |
|---|---|---|
| 1 | Lecture 1.1 — Why Graphs? | Motivation |
| 3 | Lecture 1.3 — Choice of Representation | Graph representations |
| 8 | Lecture 3.1 — Node Embeddings | Learning node representations |
| 14 | Lecture 5.1 — Message Passing | MPNN concept visually |

**After reading GCN/GAT papers (deeper GNN understanding):**

| Video # | Title | Why |
|---|---|---|
| 17-26 | Lectures 6.1–8.3 | GNN architecture, layers, design, prediction |

Key videos: **#19** (GCN explained by Leskovec) and **#23** (GAT explained).

**Optional:** #53 (Scaling GNNs), #58 (Pre-Training GNNs)

---

## 5. The Schedule

### Month 1: Foundations (May 9 – June 8, 2026) — 100% free

```
Week 1 (May 11-17): ★ CURRENT WEEK
├── Math prerequisites (daily, ongoing throughout week)
├── Read Goldberg & Levy "word2vec Explained"
├── Read Mikolov et al. "Efficient Estimation of Word Representations"
├── Read "Attention Is All You Need"
└── Read DLinear ("Are Transformers Effective for TS Forecasting?")

Week 2 (May 18-24): GNN foundations
├── Watch CS224W videos #1, #3, #8, #14 — 0.5 day
├── Read GCN (Kipf & Welling) — 1.5 days
│   └── If confused: watch CS224W video #19
├── Read MPNN (Gilmer et al.) — 1 day
├── Read GAT (Veličković et al.) — 1 day
└── Run a basic GCN example in PyTorch Geometric — 0.5 day

Week 3 (May 25-31): Time series deep learning
├── Watch Karpathy "Let's build GPT" — 1 evening
├── Read PatchTST (critical — take detailed notes) — 1.5 days
├── Read Informer, Autoformer — 2 days
├── Read BERT (skim) + TSMixer — 1.5 days
└── Set up dev environment (PyTorch, HuggingFace, GPU access)

Week 4 (Jun 1-7): Your backbone
├── READ TTM PAPER 3 TIMES — 3 days
├── Clone ibm-granite/granite-tsfm, run TTM tutorials — 2 days
├── Read Chronos, TimesFM — 1.5 days
└── Start a research notebook
```

### Month 2: Deep Dive + First Code (June 8 – July 8, 2026) — 100% free

```
Week 5 (Jun 8-14):
├── Read Moirai, Time-LLM (skim Lag-Llama, One Fits All) — 3 days
├── Run Chronos + TimesFM on traffic dataset — 2 days
└── Write 1-page: TTM vs Chronos vs TimesFM comparison

Week 6 (Jun 15-21):
├── Watch CS224W videos #17-26 (GNN depth) — 2 days
├── Read STGCN, DCRNN — 2 days
├── Run STGCN on METR-LA or PEMS-BAY — 1 day
└── Read Graph WaveNet — 0.5 day

Week 7 (Jun 22-28):
├── Read GPT-ST — 1.5 days
├── READ UniST CAREFULLY — 2 days (closest to your project)
├── Read UrbanGPT — 1.5 days
└── Identify your specific research gap

Week 8 (Jun 29-Jul 5):
├── Read Phase E papers (27-30) — 3 days
├── Design your model architecture on paper — 2 days
└── Write problem formulation + related work (~3 pages)
```

### Month 3: Build It (July 6 – August 8, 2026) — 100% free

```
Week 9-10: Implementation
├── Data pipeline for urban datasets
├── TTM fine-tuning on traffic data → baseline results
├── Implement spatial adapter (v1: fixed graph)
├── Run on 2-3 datasets, compare against STGCN, UniST, vanilla TTM
└── Debug, iterate, tune

Week 11-12: First results + Causal V2
├── Ablation studies (does V1 adapter beat vanilla TTM?)
├── Start Causal Spatial Adapter V2 (causal graph discovery / confounder mitigation)
├── Generate tables + figures
└── Identify what to improve in months 4-12
```

### Months 4-7: Harden + Extend (Aug – Nov 2026) — 70% free

```
Month 4 (Aug): Multi-domain data + heterogeneous graph
├── Collect and align multi-domain datasets from NYC:
│   ├── Traffic: NYC Taxi + Bike trip records
│   ├── Energy: Building energy usage (NYC Open Data)
│   ├── Transit: Subway ridership (MTA turnstile data)
│   └── Weather: Temperature, precipitation (NOAA)
├── Build heterogeneous urban graph:
│   ├── Nodes = traffic sensors + energy meters + transit stations
│   ├── Edges = spatial proximity + functional relationships
│   └── Different node/edge types (heterogeneous, not homogeneous)
└── Coursework begins (~30% time)

Month 5 (Sep): Multi-domain adapter + multi-task prediction
├── Extend spatial adapter to handle heterogeneous graph
│   (different node types need different embeddings)
├── Add multi-task prediction heads:
│   ├── Head 1: predict traffic flow/speed
│   ├── Head 2: predict energy consumption
│   └── Head 3: predict transit ridership
├── Key experiment: does cross-domain input improve each prediction?
│   (e.g., does knowing transit ridership help predict nearby traffic?)
└── Compare single-domain vs multi-domain performance

Month 6-7 (Oct-Nov): Exhaustive evaluation
├── Single-domain: METR-LA, PEMS-BAY, PEMS03/04/07/08 (traffic only)
├── Multi-domain: NYC (traffic + energy + transit), Chicago
├── 10+ baselines (classical + DL + foundation + ST-FM)
├── Ablation: TTM alone vs +adapter vs +multi-domain vs +multi-task
├── Efficiency analysis, sparse data experiments
├── Visualization of learned cross-domain spatial structure
└── Statistical significance: 5 seeds, mean ± std
```

### Months 8-12: Scale + Thesis (Dec 2026 – May 2027) — 70% free

```
Month 8-9 (Dec-Jan): Cross-city transfer
├── Try in order: zero-shot → multi-city training → few-shot fine-tuning
├── Scale to UTD19 (40 cities)
└── Report what actually works (even if it's the simplest approach)

Month 10-11 (Feb-Mar): Interventional Causal Forecasting + Thesis
├── Interventional analysis: "P(Traffic | do(Close Road X))"
├── Validate against real historical road closures using do-calculus
├── Write thesis (7 chapters, ~42 pages)
└── Get advisor feedback

Month 12 (Apr-May): Defense
├── Defense slides (30-40 slides)
├── Practice 3 times
├── Make GitHub repo public
└── PhD application prep (deadlines: Nov 2027 – Feb 2028)
```

---

## 6. Evaluation Checklist

> [!TIP]
> **This is what separates accepted papers from rejected ones.** Do ALL of these.

```
├── Single-domain datasets: METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
├── Multi-domain datasets: NYC (traffic+energy+transit), Chicago
├── 10+ baselines:
│   ├── Classical: HA, ARIMA, VAR
│   ├── DL: STGCN, DCRNN, Graph WaveNet, AGCRN
│   ├── Foundation: vanilla TTM, Chronos, TimesFM
│   └── ST-FM: UniST, UrbanGPT
├── Metrics: MAE, RMSE, MAPE for each horizon (15/30/60 min)
├── Ablation: TTM alone vs +adapter vs +multi-domain vs +multi-task
├── Cross-domain ablation: does energy data help traffic prediction? (and vice versa)
├── Efficiency: params, FLOPs, training time, inference time
├── Visualization: learned cross-domain spatial structure
└── Statistical significance: 5 seeds, mean ± std
```

---

## 7. Datasets

| Dataset | What | Size | Use For |
|---|---|---|---|
| **METR-LA** | LA highway traffic speed (207 sensors) | 4 months | Primary benchmark |
| **PEMS-BAY** | Bay Area traffic speed (325 sensors) | 6 months | Primary benchmark |
| **PEMS03/04/07/08** | California traffic flow | Varies | Scalability |
| **NYC Taxi/Bike** | Trip records, grid-based demand | Years | Demand prediction |
| **Chicago Traffic Tracker** | Congestion estimates | Continuous | Cross-city transfer |
| **UTD19** | 20,000+ sensors, 40 cities | Huge | Cross-city generalization |

---

## 8. Learning Resources

| Resource | What | When | Priority |
|---|---|---|---|
| **Phase 0 math videos** (Section 2) | Eigenvalues, Laplacian, Convolution | Week 1 | 🔴 Critical |
| **Karpathy "Let's build GPT"** | Transformer from scratch in code | Week 3 | 🔴 Critical |
| **CS224W — 19 videos** (Section 4) | GNN foundations | Week 2 + Week 6 | 🟡 Important |
| **HuggingFace Time Series Course** | Hands-on TS with Transformers library | Month 2 | 🔴 Critical |
| **IBM Granite TSFM Wiki + Tutorials** | Your actual backbone | Month 1 | 🔴 Critical |

### Code You Must Run

| Repository | What to Do |
|---|---|
| `ibm-granite/granite-tsfm` | Clone, run TTM inference, fine-tune on traffic data |
| `amazon-science/chronos-forecasting` | Run Chronos for comparison baseline |
| `google-research/timesfm` | Run TimesFM for comparison baseline |
| `torch_geometric` examples | Run basic GNN + ST-GNN examples |

---

## 9. Conference Deadlines

| Venue | Deadline | What to Submit |
|---|---|---|
| **WWW / ICDE 2027** | ~Oct 2026 | Paper 1: Spatial adapters + physics + multi-task |
| **KDD 2027** | ~Feb 2027 | Paper 2: Full system + cross-city at scale |
| **NeurIPS / CIKM 2027** | ~May 2027 | Paper 3 (optional): Complete system + scenarios |

> [!NOTE]
> Paper submission is NOT the priority. The priority is doing excellent work for 12 months. Submit when the work is genuinely ready.

---

## 10. PhD Application Strategy

### Target Labs

| Faculty | University | Why |
|---|---|---|
| Hui Xiong | HKUST | Co-author of UFM survey, leads UFM research |
| Yuxuan Liang | HKUST(GZ) | UniST author, spatio-temporal FMs |
| Junbo Zhang | JD / Tsinghua | Urban computing pioneer |
| Licia Capra | UCL | Urban computing + ML |
| Cyrus Shahabi | USC | Spatial data + deep learning |
| Roger Zimmermann | NUS | Urban computing + multimedia |

---

## 11. Mistakes to Avoid

1. **Reading too broadly, building nothing.** Start coding by Week 5.
2. **Stacking techniques for show.** One idea done well > five ideas done poorly.
3. **Not running baselines early.** Get STGCN/UniST running by Week 6.
4. **Ignoring reproducibility.** Seed everything, log experiments (Weights & Biases).
5. **Writing the thesis last.** Start related work in Week 7.
6. **Not talking to your advisor.** Share your plan by end of Week 2.

---

## 12. Errata Log

| Issue | Status | Detail |
|---|---|---|
| Missing GNN papers (GCN, GAT, MPNN) | ✅ Fixed | Added as Phase A.5 |
| Rusek & Chołda "MPNN Learn Little's Law" | ✅ Clarified | Computer networking paper, NOT urban traffic. Not needed. |
| TTM architecture | ✅ Verified | MLP-Mixer-based, NOT Transformer. 1-5M params. NeurIPS 2024. |
| UniST architecture | ✅ Verified | Transformer encoder-decoder + ST patching + prompt tuning. KDD 2024. |
| ViT, TSMixer redundancy | ✅ Removed | PatchTST covers patching concept. TSMixer covered via TTM paper. |
| Survey paper | ✅ Changed | Read sections as needed, not cover-to-cover upfront. |
| Physics constraints | ✅ Deprioritized | Replaced by multi-domain integration as Month 4-7 focus. May still add as a minor experiment. |
| Multi-domain integration | ✅ Added | Heterogeneous urban graph (traffic+energy+transit) is now the core Month 4-7 contribution. |
| Optimal Transport | ✅ Clarified | Try simple cross-city approaches first. OT only if they plateau. |
| "Counterfactual reasoning" | ✅ Renamed | It's perturbation-based scenario analysis, not formal causal inference. |
