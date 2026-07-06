# Missing Paradigms Assessment — Are We Missing Anything?

> Assessed on June 5, 2026. Covers RL, diffusion models, graph transformers, self-supervised learning, and more.

---

## ❌ Not Missing — Would Hurt You

| Paradigm | Why it doesn't fit |
|---|---|
| **Reinforcement Learning** | RL answers "what action should I take?" (e.g., change traffic signal to green). Your project answers "what will happen?" These are fundamentally different problems. Adding RL would double your scope for zero thesis coherence. |
| **Diffusion Models for time-series** | Trendy, but they're mainly for generation/imputation, not forecasting. Your problem is prediction, not synthesis. |
| **Multi-agent simulation** | Modeling individual cars as agents is a different field entirely (SUMO, VISSIM). Your model works at the sensor level, not vehicle level. |
| **Federated Learning** | Great "future work" paragraph in your thesis. Terrible idea to actually implement in 12 months alongside everything else. |

---

## 🟡 Not Missing, But Worth Knowing About

| Paradigm | Verdict |
|---|---|
| **Physics-Informed constraints** | Your roadmap already considered and correctly deprioritized this. Traffic flow follows conservation laws (flow = density × speed). Adding a physics loss term is ~1 day of work and could be a small experiment in your thesis. Don't build your project around it. |
| **LLM-as-reasoner** | UrbanGPT already does this. Could be a cool demo layer ("Why is there congestion on I-405?") but it's a presentation trick, not a research contribution. Save for the PhD. |

---

## ✅ Actually Missing — Worth Considering

### 1. Graph Transformers (instead of GCN in your adapter)

Your plan uses GCN for the spatial adapter. But **Graph Transformers** (Graphormer, GPS) are now state-of-the-art for many graph tasks. The key difference:

```
GCN:              Each node only sees immediate neighbors (1-hop)
Graph Transformer: Each node attends to ALL nodes, with structural bias
```

For traffic, this matters — a highway closure 10 km upstream affects you, but GCN with 1-2 layers can't "see" that far. A Graph Transformer can.

> **Practical suggestion:** Use GCN for V1 (it's simpler). If results plateau, swap in a Graph Transformer for V2. This becomes a nice ablation: "GCN adapter vs. Graph Transformer adapter."

### 2. Self-supervised pre-training for the adapter

Right now your plan is: build adapter → train it on labeled traffic data. But what if you first **pre-train the adapter** with a self-supervised objective?

```
Step 1: Mask 15% of sensors, ask the adapter to reconstruct them
        from neighbors (no labels needed — just graph structure)
Step 2: Fine-tune the pre-trained adapter on traffic prediction
```

This is exactly what BERT does for language. It could mean your adapter needs less labeled data to work well — which directly strengthens your cross-city transfer story (Objective 5).

> **Effort:** ~1 week. **Impact:** Could be a standalone contribution.

### 3. Temporal Graph Networks

Your plan treats the graph as **static** (the road network doesn't change). But traffic relationships ARE dynamic — Sensor A influences Sensor B during rush hour but not at 3 AM. **Temporal edges** that change weight over time could capture this.

> **Practical suggestion:** Start with a static graph. Then try a simple time-varying version: morning graph, afternoon graph, night graph (3 adjacency matrices). If that helps, you've found something interesting.

---

## The Bottom Line

Your plan covers the right paradigms. The things genuinely worth adding are:

1. **Graph Transformer as adapter alternative** — easy to swap in later
2. **Self-supervised adapter pre-training** — strengthens cross-city transfer
3. **Time-varying adjacency** — simple to try, potentially impactful

Everything else (RL, diffusion, federated, multi-agent) belongs in a PhD, not this thesis. **Depth beats breadth.**
