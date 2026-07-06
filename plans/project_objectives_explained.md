# M.Tech Project: Six Objectives Explained in Detail

---

## Objective 1: Build a Spatial Adapter for Frozen TTM (Months 1-3)

### The Problem

IBM's Tiny Time Mixers (TTM) is a powerful time-series prediction model. It was pre-trained on **billions** of time-series data points from diverse domains (finance, weather, retail, etc.). Because of this massive pre-training, it can predict the future values of almost any time-series with good accuracy, even without domain-specific training.

However, TTM has a critical blind spot: **it treats every sensor as an independent, isolated entity.**

Imagine a city with 200 traffic sensors:
```
Sensor #1:  [60, 58, 55, 50, 45, ...] km/h  → TTM predicts: [40, 38, ...]
Sensor #2:  [70, 70, 68, 65, 60, ...] km/h  → TTM predicts: [55, 52, ...]
Sensor #3:  [80, 80, 80, 78, 75, ...] km/h  → TTM predicts: [72, 70, ...]
```

TTM processes each sensor separately. It sees Sensor #1 slowing down and predicts it will keep slowing down. But it has **no idea** that Sensor #1 is on a highway 500 meters upstream of Sensor #3. In reality, the jam at Sensor #1 will reach Sensor #3 in about 5 minutes. TTM cannot predict this because it doesn't know the road network exists.

### The Solution: Your Spatial Adapter

You will design a small, lightweight neural network module (the "spatial adapter") that sits between the raw data and TTM. Its job is to inject **spatial context** — information about which sensors are neighbors and how they influence each other.

**How it works conceptually:**

```
Step 1: Build a Graph
    - Each sensor = a node
    - Two sensors are connected by an edge if they are physically close
      (e.g., on the same road, within 2 km)
    - The edge has a weight proportional to how close they are

Step 2: Message Passing (before TTM sees the data)
    - For each sensor, the adapter "asks" its neighbors:
      "What are your current readings?"
    - It then creates an enriched representation:
      Sensor #3's input = [its own data] + [weighted average of neighbors' data]

Step 3: Feed into Frozen TTM
    - TTM now sees not just Sensor #3's history,
      but a spatially-enriched version that encodes the neighborhood
    - TTM is completely frozen — you do NOT change any of its weights
    - Only the adapter's weights (~100K parameters) are trained
```

**The mathematical intuition:**

The adapter performs a Graph Convolution. In its simplest form, this is:

```
H' = σ( Ã · H · W )
```

Where:
- **H** = the matrix of all sensor readings (each row is a sensor, each column is a time step)
- **Ã** = the normalized adjacency matrix of the road network (who is connected to whom)
- **W** = learnable weight matrix (these are the ONLY weights you train)
- **σ** = activation function (e.g., ReLU)
- **H'** = spatially-enriched sensor readings → this is what TTM receives

**The Causal Upgrade (V2 Adapter):**
While V1 uses standard physical distance, V2 of your adapter will be a **Causal Spatial Adapter**. Instead of assuming symmetric physical connections, it will use causal discovery (e.g., PCMCI) or confounder mitigation techniques to learn the *true, asymmetric causal graph* of traffic flow. TTM will then receive causally-enriched data.

**Why "adapter" instead of just building a new model?**

Because TTM was pre-trained on billions of data points. That knowledge (understanding trends, seasonality, sudden changes) is extremely valuable. If you built a model from scratch, you would need millions of traffic data points to learn what TTM already knows. By freezing TTM and just adding a small adapter, you get the best of both worlds:
- TTM's pre-trained temporal knowledge (free)
- Your adapter's spatial knowledge (learned from a small amount of traffic data)

This is the same principle behind LoRA adapters for large language models — people don't retrain GPT-4, they add tiny adapters.

---

## Objective 2: Beat Baselines (Month 3)

### What this means

You need to prove that your approach (TTM + Spatial Adapter) is actually better than the alternatives. In academic research, you cannot just say "my model works." You must show **numerical evidence** that it outperforms existing methods.

### The baselines you will compare against

**Category 1: Classical Methods (the weakest baselines)**
- **Historical Average (HA):** Just predict that tomorrow's traffic will be the same as the average of the last 7 days. Surprisingly hard to beat.
- **ARIMA:** A statistical model from the 1970s that looks at trends and seasonality.
- **VAR (Vector Autoregression):** Like ARIMA but considers multiple sensors at once.

**Category 2: Deep Learning Spatio-Temporal Models (the strong baselines)**
- **STGCN (2018):** The first paper to combine Graph Convolutions with temporal convolutions for traffic. This is the "classic" ST-GNN.
- **DCRNN (2018):** Models traffic as a diffusion process (like ink spreading in water) on the road graph.
- **Graph WaveNet (2019):** Learns the graph structure from data instead of using the physical road network.

**Category 3: Time Series Foundation Models (your direct competitors)**
- **Vanilla TTM (no adapter):** TTM without your spatial adapter. This is your most important baseline — you must show that adding the adapter actually helps.
- **Chronos (Amazon):** Tokenizes time-series like language.
- **TimesFM (Google):** A decoder-only foundation model for time series.

**Category 4: Spatio-Temporal Foundation Models (the hardest to beat)**
- **UniST (KDD 2024):** The closest existing work to what you're building. It also tries to be a unified spatio-temporal model.
- **UrbanGPT (KDD 2024):** Uses LLM reprogramming for urban prediction.

### How you measure "better"

You will report three standard metrics for traffic prediction:

| Metric | What it measures |
|---|---|
| **MAE** (Mean Absolute Error) | On average, how many km/h off is your prediction? |
| **RMSE** (Root Mean Square Error) | Same as MAE but punishes big errors more heavily |
| **MAPE** (Mean Absolute % Error) | What percentage off is your prediction? |

You report these for three prediction horizons: **15 minutes, 30 minutes, and 60 minutes** into the future. Predicting 15 minutes ahead is easy; 60 minutes is very hard.

### What "winning" looks like

You don't need to beat every single baseline on every single metric. A realistic, publishable result would be:

```
✅ TTM + Adapter beats vanilla TTM on all metrics
   (proves the adapter helps)

✅ TTM + Adapter matches or beats STGCN/DCRNN
   (proves you're competitive with purpose-built models)

✅ TTM + Adapter is close to UniST but uses 10x fewer parameters
   (proves efficiency advantage of adapter approach)
```

---

## Objective 3: Heterogeneous Urban Graph (Months 4-5)

### The Problem

Objective 1 uses a simple, single-domain graph: traffic sensors connected by roads. But a real city is not just roads. A city has:

- **Traffic sensors** measuring vehicle speed/flow
- **Energy meters** measuring building electricity consumption
- **Transit stations** measuring subway/bus ridership
- **Weather stations** measuring temperature and rain

These systems are deeply interconnected. When it rains, energy consumption goes up (people turn on heaters/AC), subway ridership increases (people avoid driving), and traffic slows down. A model that only looks at traffic sensors misses all of this.

### The Solution: Heterogeneous Graph

Instead of a graph where all nodes are the same type (traffic sensors), you build a graph with **multiple types of nodes and edges**:

```
Node Types:
├── Type 1: Traffic sensor (features: speed, flow, occupancy)
├── Type 2: Energy meter   (features: kWh consumption, building type)
├── Type 3: Transit station (features: hourly entries, exits)
└── Type 4: Weather station (features: temperature, precipitation)

Edge Types:
├── traffic-to-traffic: Two sensors on the same road
├── transit-to-traffic: Subway station near a traffic sensor
├── energy-to-traffic:  Building cluster near a road
└── weather-to-all:     Weather affects everything within its radius
```

**Why this is novel:**
No existing Time-Series Foundation Model handles cross-domain urban data. TTM sees numbers — it doesn't know if a number represents traffic speed or electricity usage. Your heterogeneous adapter teaches TTM that these different data types are related and influence each other.

### The Technical Challenge

Different node types have different feature dimensions (a traffic sensor has 3 features; an energy meter might have 5). You need **type-specific embedding layers** so that all node types get projected into the same vector space before the graph convolution happens:

```
Traffic sensor → Linear(3, 64) → 64-dim vector
Energy meter   → Linear(5, 64) → 64-dim vector
Transit station → Linear(2, 64) → 64-dim vector
```

Now all nodes live in the same 64-dimensional space and can "talk" to each other through the graph.

---

## Objective 4: Cross-Domain Improvement (Months 5-7)

### The Core Question

This is the most exciting scientific question in your thesis:

> **"Does knowing the subway ridership at a station help you predict traffic congestion on the nearby road?"**

If the answer is yes, then your multi-domain model is genuinely useful and novel. If the answer is no, then there's no point in building the heterogeneous graph.

### How you test this

You run a carefully designed **ablation study** — a series of experiments where you systematically add and remove components to see what helps:

```
Experiment A: Traffic data only        → MAE = X₁
Experiment B: Traffic + Weather        → MAE = X₂
Experiment C: Traffic + Transit        → MAE = X₃
Experiment D: Traffic + Energy         → MAE = X₄
Experiment E: Traffic + ALL domains    → MAE = X₅
```

**If X₅ < X₁** (i.e., using all domains gives lower error than traffic alone), you have strong evidence that cross-domain signals help. You then report exactly how much each domain contributes.

### Why this matters for publication

This is the kind of result that makes reviewers at KDD/WWW excited. It's not just "we built a bigger model." It's a scientifically falsifiable claim: **"Urban domains are interconnected, and a model that captures these connections makes better predictions."** You either prove it or disprove it — both outcomes are publishable.

### What the multi-task prediction heads look like

Your model doesn't just predict traffic. It predicts everything simultaneously:

```
Shared Backbone: Frozen TTM + Heterogeneous Spatial Adapter
                    |
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Head 1:      Head 2:      Head 3:
   Traffic      Energy       Transit
   (speed)      (kWh)        (ridership)
```

Each "head" is just a small linear layer that takes the shared representation and outputs domain-specific predictions. The key insight is that the shared backbone learns a unified urban representation, and each head specializes in one prediction task.

---

## Objective 5: Cross-City Transfer (Months 8-9)

### The Problem

Your model is trained on Los Angeles traffic data. But can it predict traffic in San Francisco? In New York? In Mumbai?

Every existing traffic model is trained from scratch for each city. If a transportation agency in a new city wants to use deep learning for traffic prediction, they need:
- Months of historical sensor data from their specific city
- GPUs and engineering expertise to train a model
- Weeks of hyperparameter tuning

This is extremely expensive and impractical for most cities in the world.

### The Solution: Foundation Model Transfer

Because TTM was pre-trained on billions of time-series (not just traffic), it already understands general patterns like trends, seasonality, and sudden changes. Your spatial adapter learns how road networks affect traffic flow. The hypothesis is that **road network effects are similar across cities** — a traffic jam spreading through an intersection in LA works the same way as in Tokyo.

### How you test this

You will try three approaches in order of difficulty:

```
Approach 1: Zero-shot
├── Train on LA
├── Test on Bay Area WITHOUT any fine-tuning
└── Question: Does it work at all?

Approach 2: Multi-city training
├── Train on LA + Bay Area + NYC simultaneously
├── Test on Chicago (a city it has never seen)
└── Question: Does training on more cities help generalization?

Approach 3: Few-shot fine-tuning
├── Train on LA
├── Fine-tune on just 1 week of Chicago data
├── Test on the remaining Chicago data
└── Question: How little data from a new city do you need?
```

### The Scale

The **UTD19 dataset** contains traffic data from **40+ cities** across 20 countries. If your model works across even 10 of these cities, you have a very strong paper. No existing spatio-temporal model has demonstrated this kind of cross-city generalization at scale.

---

## Objective 6: Interventional Causal Forecasting (Month 10)

### The Problem

City planners don't just want to predict traffic — they want to answer **causal "what if" questions**:
- "What happens to traffic if we intervene and close Main Street?"
- "What happens to the subway if we add a new bus route?"

Standard deep learning models cannot answer these questions because they only learn observational correlations ($P(Y|X)$). If you change the underlying system, the correlation breaks.

### The Solution: do-calculus and Interventional Forecasting

Because your V2 model uses a **Causal Spatial Adapter**, you have access to the true causal graph. You can perform formal causal inference using Judea Pearl's **do-calculus**:

```
Normal observational prediction:
├── P(Traffic | History)

Causal Intervention: "Close Road X"
├── P(Traffic | do(Road X = closed))
├── Mathematically cut the incoming edges to Road X in your causal graph
├── Run the model again to simulate the intervention
├── Quantify the true causal effect on downstream roads
```

### How you validate this

You look for **real historical events** where a road was actually closed (e.g., for construction or an accident) and check whether your model's simulated scenario matches what actually happened in the real data.

```
Historical event: Road X was closed on March 15, 2023
Real data shows:  Road Y traffic increased by 25%
Your simulation:  Road Y traffic predicted to increase by 22%
```

If these numbers are close, your scenario analysis is validated.

### Why this is valuable

This transforms your model from a **passive prediction tool** into an **active planning tool**. City planners can use it to evaluate infrastructure changes before they happen. This is the kind of practical impact that makes PhD applications stand out.

---

## Summary: The Six Objectives as a Pyramid

```
                    ┌──────────────────┐
                    │  6. Scenario     │  ← "What if Road X closes?"
                    │     Analysis     │
                    ├──────────────────┤
                 ┌──┤  5. Cross-City   │  ← Train on LA, predict in Tokyo
                 │  │     Transfer     │
                 │  ├──────────────────┤
              ┌──┤  │  4. Cross-Domain │  ← Does subway data help traffic?
              │  │  │     Proof        │
              │  │  ├──────────────────┤
           ┌──┤  │  │  3. Heterogeneous│  ← Traffic + Energy + Transit graph
           │  │  │  │     Urban Graph  │
           │  │  │  ├──────────────────┤
        ┌──┤  │  │  │  2. Beat         │  ← Prove it works (MAE, RMSE, MAPE)
        │  │  │  │  │     Baselines     │
        │  │  │  │  ├──────────────────┤
        │  │  │  │  │  1. Causal        │  ← The core: causal adapter + frozen TTM
        │  │  │  │  │     Adapter       │
        └──┴──┴──┴──┴──────────────────┘

Objectives 1-2 = Solid M.Tech thesis
Objectives 1-4 = Strong publication (KDD/WWW)
Objectives 1-6 = World-class Causal AI PhD application
```

---

## 7. Required Reading per Objective (Core vs. Literature Review)

There are 30+ papers in the master roadmap, but you do **not** read them all deeply. You deeply read the **Core Engineering Papers** (to write the code) and skim the **Literature Review Papers** (to cite in your thesis).

### Objective 1: Spatial Adapter for Frozen TTM
**The Book Text (Read deeply before the papers):**
*   *Jurafsky & Martin:* Transformer Chapter (for Attention intuition).
*   *Yao Ma "Deep Learning on Graphs":* Chapters 5, 6, and 12 (for GCNs and Spatio-Temporal math).

**Core Engineering Papers (Read deeply):**
*   *GCN (Kipf & Welling)* — The math for the spatial graph.
*   *STGCN* — How to combine graphs with time series.
*   *PatchTST* — How Transformers digest time-series data.
*   *TTM (IBM)* — Your actual backbone model.

### Objective 2: Beat Baselines
**Core Engineering Papers (Read deeply):**
*   *UniST (KDD 2024)* — The closest existing model to yours. You must understand exactly how it works so you can beat it.

**Literature Review Papers (Skim the abstract, diagrams, and results to cite them):**
*   *DCRNN, Graph WaveNet* (Graph competitors)
*   *Chronos, TimesFM* (Transformer competitors)

### Objective 3 & 4: Heterogeneous Graph & Cross-Domain Proof
**Core Engineering Papers (Read deeply):**
*   *BIGCity* — The latest attempt at a unified urban model.
*   *UrbanCLIP* — Handling multimodal urban data.

**Literature Review Papers (Skim to cite):**
*   *One Fits All, STD-PLM*

### Objective 5: Cross-City Transfer
**Core Engineering Papers (Read deeply):**
*   *Cross-City Transfer Learning for Deep Spatio-Temporal Prediction (IJCAI 2018)* — The benchmark methodology.
*   *UTD19 Dataset Paper* — Understand the data structure of the 40 cities.

### Objective 6: Interventional Causal Forecasting
**Core Engineering Papers (Read deeply):**
*   *CISTGNN (2024)* — How to remove hidden confounders and build a true causal STGNN.
*   *Chris Olah's Blog / Judea Pearl basics* — For `do-calculus` intuition.

*(Note: Papers like Word2Vec, BERT, TS2Vec, Lag-Llama, etc., from your tracker are purely historical context or literature review. Skim them when writing your thesis chapter 2, do not implement them.)*
