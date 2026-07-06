# Honest Assessment of Your M.Tech Project Plan

> A critical review with concrete suggestions. No sugarcoating.

---

## Part 1: What's Genuinely Strong

Let me start with what the plan gets right, because these are real strengths:

| Strength | Why it matters |
|---|---|
| **The adapter-over-foundation-model idea** | This is a genuinely good research direction. It mirrors what LoRA did for LLMs. The insight that you can inject spatial structure without retraining the backbone is clean and publishable. |
| **Clear pyramid of objectives** | The 1→6 layering means you can stop at any point and still have a thesis. This is smart project design. |
| **Strong baseline list** | Comparing against 10+ baselines across 4 categories (classical, DL, foundation, ST-FM) is thorough. Reviewers will appreciate this. |
| **The roadmap document** | Honestly one of the most well-structured M.Tech plans I've seen. The reading order, time estimates, and phase structure are excellent. |
| **PhD-oriented framing** | Targeting KDD/WWW/NeurIPS and listing specific faculty shows serious intent. |

---

## Part 2: Honest Concerns

### 🔴 Concern 1: The Plan is Too Ambitious for 12 Months

This is the biggest risk. Let me be blunt:

```
Objectives 1-2:  Solid M.Tech thesis         ← achievable in 12 months
Objectives 1-4:  KDD paper                   ← tight but possible
Objectives 1-6:  "World-class PhD material"   ← extremely unlikely in 12 months
```

**The reality check:** You are currently in Week 4 of a 12-month plan. Your `src/` directory is empty — no code has been written yet. You have spent 4 weeks reading papers and math, which is good, but you are now entering the phase where things slow down dramatically:

- **Month 3 expects**: Data pipeline + TTM fine-tuning + Spatial adapter v1 + Run on 2-3 datasets + Compare against STGCN, UniST, vanilla TTM + Debug + Start Causal V2 + Ablation studies + Generate tables and figures.
- **That is 4-6 months of work compressed into 4 weeks.** Building a data pipeline alone (downloading METR-LA, cleaning it, writing PyTorch Dataset classes, handling temporal splits) takes a full week for someone who hasn't done it before.

> [!WARNING]
> Months 4-12 assume 70% free time due to courses. That's optimistic. With IISc coursework (assignments, mid-terms, projects, end-sems), 50% is more realistic. That means Objectives 4-6 each get roughly **2-3 weeks of actual work time**, not months.

**My recommendation:**
- **Guaranteed deliverable:** Objectives 1-3 (adapter + baselines + heterogeneous graph). This alone is a strong M.Tech thesis.
- **Stretch goal:** Objective 4 (cross-domain proof). If you get this, you have a paper.
- **Bonus:** Objectives 5-6. Attempt only if everything before goes smoothly. Do NOT plan your thesis around these.

---

### 🔴 Concern 2: The Pre-Adapter Architecture May Not Work Well Enough

Your [technical blueprint](file:///Users/swapnilaggarwal/M.Tech%20Project/objective1_technical_blueprint.md) describes a Pre-Adapter: do graph convolution first, then feed the result to TTM. This is the simplest approach, but it has a fundamental limitation:

```
Current Plan:
  Raw Data → [GCN Adapter] → Spatially-mixed Data → [Frozen TTM] → Prediction
                                                      ↑
                                            TTM sees modified numbers
                                            but has NO knowledge that
                                            a graph exists
```

**The problem:** TTM was pre-trained on raw time-series. When you spatially mix the data before feeding it in, you're giving TTM something that looks *different* from what it was trained on. The pre-training distribution assumed each channel is independent. By mixing channels through graph convolution, you may actually *confuse* TTM's pre-trained representations rather than help them.

**This is not hypothetical.** The DLinear paper showed that even simple linear models can beat Transformers on time-series when the Transformer is given data in the wrong format. Distribution shift between pre-training and fine-tuning is a known failure mode.

> [!IMPORTANT]
> **What to do about this:** You MUST test the Pre-Adapter against a second architecture early. I'd suggest also implementing a **Post-Adapter** (TTM processes each sensor independently → GCN mixes the per-sensor predictions → final prediction). If Post-Adapter beats Pre-Adapter, that's actually a publishable finding ("spatial context is more effective at the prediction stage than the input stage").

**Even better — try a Parallel-Adapter:**
```
Raw Data ──→ [Frozen TTM] ──→ Temporal Features ──┐
    │                                              ├──→ Fusion → Prediction
    └──→ [GCN Adapter] ──→ Spatial Features ──────┘
```
This avoids the distribution shift problem entirely. TTM sees clean data it was trained on, the GCN sees the graph structure, and a small fusion layer combines both.

---

### 🟡 Concern 3: Your Novelty Claim May Be Weaker Than You Think

The core claim is: *"Nobody has designed causal adapters for TS foundation models."*

This was probably true when the plan was written (May 2026). But the field moves fast:

- **Adapter-for-foundation-models** is now a crowded space in NLP (LoRA, QLoRA, AdaLoRA, etc.) and is rapidly being adopted in time-series.
- By the time you submit to WWW (Oct 2026 deadline), there may be 2-3 papers doing "spatial adapters for time-series foundation models" — this exact idea is "in the air."
- UniST and UrbanGPT (both KDD 2024) are already doing spatial + foundation model combinations, just with different architectures.

> [!IMPORTANT]
> **What makes your work genuinely novel is NOT the adapter itself — it's the multi-domain heterogeneous graph (Objective 3) and the causal graph (Objective 6).** If you only do Objectives 1-2 with a standard distance-based adjacency matrix, you risk being scooped.
>
> **Priority shift:** Don't spend 3 months perfecting the basic adapter. Get it working in 4-6 weeks, then invest your time in the heterogeneous graph and cross-domain proof. That's where the real novelty lives.

---

### 🟡 Concern 4: Missing Pieces in the Technical Blueprint

The [blueprint](file:///Users/swapnilaggarwal/M.Tech%20Project/objective1_technical_blueprint.md) has several gaps:

1. **No data splitting strategy.** Traffic data is temporal — you can't do random train/test splits. You need chronological splits (e.g., first 70% for training, next 10% for validation, last 20% for test). This is a common mistake in ST papers.

2. **No normalization/standardization.** Traffic speed ranges vary across sensors. You need Z-score normalization per sensor. When you compute MAE, you compute it on de-normalized values.

3. **No multi-horizon prediction.** The blueprint only mentions single-step prediction. Your evaluation requires 15/30/60-minute horizons (3/6/12 steps ahead for 5-min data).

4. **No attention to TTM's actual API.** TTM (IBM Granite) has specific input format requirements. The blueprint uses a generic `TTMModel.from_pretrained()` call that doesn't match the actual HuggingFace API. You need to study the `granite-tsfm` codebase before writing the adapter.

5. **The `torch.matmul(Adjacency_Matrix, X)` line is dimensionally incorrect** for the stated shape `(Batch, N, Context_Length, Features)`. Adjacency matrix is `[N, N]` — you'd need `torch.einsum('ij,bjkl->bikl', A, X)` or explicit reshaping.

---

### 🟡 Concern 5: The Causal Claims Are the Riskiest Part

Objective 6 (Interventional Causal Forecasting) sounds impressive but is the most likely to fail:

- **PCMCI for causal discovery** requires stationarity assumptions that traffic data violates (traffic patterns are very different at 8 AM vs 3 AM).
- **do-calculus validation** against real road closures is conceptually clean but practically very hard. You need to find historical road closures, get ground-truth data showing the downstream effect, and show your model predicts it. This requires very specific data that may not exist in METR-LA or PEMS-BAY.
- **Reviewers will be skeptical.** Causal claims in ML papers are scrutinized heavily. If your causal graph is just a slightly modified adjacency matrix, reviewers will argue it's not truly causal.

**My honest take:** Objective 6 is best positioned as a "future work" section or a small exploratory experiment (2-3 pages in the thesis), not a core contribution. Don't promise "Causal AI" in your thesis title unless you can deliver rigorous causal analysis.

---

## Part 3: What You Can Do That's Even More Impactful

Here are concrete ideas to make this project significantly stronger, ordered by effort-to-impact ratio:

### Idea 1: Build an Interactive Demo (High Impact, Medium Effort)

**What:** A web dashboard where someone can:
- Select a city (LA, Bay Area, NYC)
- See real-time traffic predictions vs. actual
- Toggle domains on/off (traffic only → traffic + transit → all domains)
- Visually see how cross-domain signals improve predictions

**Why this matters:**
- City planners can't read MAE tables. A visual demo makes your work **tangible**.
- PhD applications with live demos stand out dramatically.
- Conference reviewers increasingly expect demos or supplementary videos.
- This is something I can help you build directly.

**Effort:** ~1 week of coding once you have trained models.

---

### Idea 2: Uncertainty Quantification (High Impact, Low Effort)

**What:** Instead of predicting "speed = 40 km/h", predict "speed = 40 ± 5 km/h with 95% confidence."

**Why this matters:**
- City planners need to know **how confident** the prediction is. A prediction of "speed = 40 km/h" is useless if the model is wildly uncertain.
- Almost no existing ST-GNN paper reports uncertainty. This is a genuine gap.
- It's relatively easy to add — you can use **Monte Carlo Dropout** (just add dropout to your adapter and run inference 20 times) or **conformal prediction** (a post-hoc method that requires zero architecture changes).

**Effort:** ~2-3 days on top of existing model.

---

### Idea 3: Attention Visualization / Explainability (High Impact, Low Effort)

**What:** Visualize which sensors influence which. Show heatmaps of learned spatial attention weights overlaid on a map of LA/Bay Area.

**Why this matters:**
- If your model learns that Sensor #42 is heavily influenced by Sensor #17 (even though they're not physically adjacent), that's a **discovery** — it means there's a hidden traffic flow pattern.
- Explainability is a hot topic. Adding one figure showing "what the model learned" makes your thesis 10x more interesting.
- Reviewers at KDD love interpretable visualizations.

**How:** If you use GAT (attention-based) instead of plain GCN in your adapter, you get attention weights for free. Even with GCN, you can visualize the learned weight matrix.

**Effort:** ~2-3 days.

---

### Idea 4: Real-Time Streaming Inference (Medium Impact, Medium Effort)

**What:** Instead of batch prediction on historical data, show the model running on a **live data stream** (or simulated live stream).

**Why this matters:**
- Every existing paper evaluates on static historical datasets. Showing your model works in a streaming setting is practically novel.
- This is a natural requirement for actual deployment (city traffic centers need real-time predictions).
- Connects naturally to the Edge AI course in your curriculum.

**Effort:** ~1 week.

---

### Idea 5: Fairness Analysis (Low-Hanging Fruit, High Novelty)

**What:** Does your model predict equally well for all neighborhoods? Or does it work great in downtown LA (dense sensor coverage) and poorly in outer suburbs (sparse sensors)?

**Why this matters:**
- **Zero** ST-GNN papers analyze fairness. This is a genuine gap.
- If you can show that your adapter reduces the performance gap between dense and sparse regions (because graph convolution borrows information from neighbors), that's a publishable finding on its own.
- Extremely relevant for "Smart Cities" framing — you don't want AI that only works in rich neighborhoods.

**Effort:** ~1-2 days of analysis on existing results.

---

### Idea 6: Lightweight Adapter Efficiency Story (Medium Impact, Low Effort)

**What:** Don't just show your model is accurate — show it's **efficient**. Report:
- Adapter parameters vs. full model parameters (e.g., "100K adapter params vs. 5M TTM params = 2% overhead")
- Training time: adapter fine-tuning vs. training STGCN from scratch
- Inference latency comparison
- Memory footprint

**Why this matters:**
- If your adapter achieves 95% of UniST's performance with 10x fewer trainable parameters and 5x faster training, that's a STRONGER result than beating UniST by 0.5% MAE.
- Efficiency is increasingly valued at top venues. Papers like DLinear succeeded precisely because "simple + efficient > complex + marginal gain."

**Effort:** ~1 day — just measure and report what you already have.

---

## Part 4: Revised Priority Recommendation

If I were your advisor, here's what I'd recommend:

### Must-Do (Core Thesis)
| # | What | Timeline |
|---|---|---|
| 1 | **Spatial Adapter V1** (try Pre, Post, and Parallel) | Weeks 9-11 |
| 2 | **Beat baselines** on METR-LA and PEMS-BAY | Week 12-13 |
| 3 | **Heterogeneous Urban Graph** (traffic + transit + weather) | Month 4-5 |
| 4 | **Cross-domain ablation** (does adding transit help traffic?) | Month 5-6 |
| 5 | **Efficiency analysis** (params, FLOPs, training time) | Month 6 |
| 6 | **Explainability visualizations** | Month 6-7 |

### Should-Do (Paper-Worthy)
| # | What | Timeline |
|---|---|---|
| 7 | **Uncertainty quantification** (MC Dropout or conformal) | Month 7 |
| 8 | **Fairness analysis** across sensor density | Month 7 |
| 9 | **Cross-city transfer** (zero-shot + few-shot, 3-5 cities) | Month 8-9 |
| 10 | **Interactive demo dashboard** | Month 9-10 |

### Nice-to-Have (Thesis Extras)
| # | What | Timeline |
|---|---|---|
| 11 | Causal graph discovery (PCMCI or Granger) | Month 10-11 |
| 12 | Interventional scenarios | Month 11 |
| 13 | Streaming inference demo | Month 11-12 |

---

## Part 5: The One Change That Would Make the Biggest Difference

> [!CAUTION]
> **Start coding THIS WEEK.**
>
> You've been in reading mode for 4 weeks. The reading is important, but you're entering the danger zone of "reading too broadly, building nothing" — the #1 mistake your own roadmap warns against.
>
> Here's what I'd do this week:
> 1. Download METR-LA dataset (30 minutes)
> 2. Write a PyTorch Dataset class for it (2 hours)
> 3. Run vanilla TTM inference on it (1 day — will involve fighting with the HuggingFace API)
> 4. Get a baseline MAE number
>
> Once you have that number, everything else becomes "can I beat this number?" That's a much more motivating structure than "read paper #17."

---

## Summary

| Aspect | Grade | Comment |
|---|---|---|
| Research direction | **A** | Genuine gap, timely topic, strong framing |
| Plan structure | **A** | Exceptionally well-organized |
| Timeline realism | **C+** | 6 objectives in 12 months is too aggressive |
| Technical depth | **B-** | Blueprint has gaps; adapter architecture needs alternatives |
| Novelty protection | **B** | Core idea may be scooped; heterogeneous graph is the real novelty |
| Missing elements | **C** | No uncertainty, no explainability, no fairness, no efficiency story |
| Execution progress | **D** | 4 weeks in, src/ is empty |

**The plan is excellent on paper. The risk is not that the plan is bad — it's that it's so ambitious that partial execution will feel like failure. Scope it down to 3-4 core objectives, add the low-hanging impactful ideas (uncertainty, explainability, fairness, efficiency), and start coding immediately.**
