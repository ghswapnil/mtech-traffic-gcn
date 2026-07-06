# TTM + GCN: Enhancing Time-Series Foundation Models with Spatial Adapters

This repository contains the codebase for the M.Tech Thesis: **"Enhancing Time-Series Foundation Models with Spatial Graph Adapters for Traffic Forecasting"**. 

We propose a lightweight architecture that augments a frozen [Tiny Time Mixer (TTM)](https://github.com/ibm-granite/granite-tsfm) temporal backbone with a trainable Graph Convolutional Network (GCN) branch. This allows the model to capture spatial dependencies across highway sensor networks without needing to retrain the massive temporal foundation model.

## Results
Our best model, the **Co-Fine-Tuned Head + Adapter** (Model 5), achieves a **33.5% reduction** in overall MAE compared to the zero-shot TTM baseline on the METR-LA dataset. 

| Horizon | MAE | RMSE | MAPE |
|---------|-----|------|------|
| 15 min  | 3.52 | 6.57 | 9.5% |
| 30 min  | 4.07 | 7.69 | 11.7% |
| 60 min  | 4.75 | 8.86 | 14.8% |

Notably, our 60-minute RMSE of **8.86** surpasses fully-trained spatial-temporal models like STGCN, while requiring only **156K trainable parameters**.

## Project Structure
- `src/`: Source code for data loading, graph creation, and training scripts.
  - `models/`: Contains the PyTorch neural network architectures (`adapter.py`, `gcn.py`).
  - `model5_co_finetune.py`: Script to train our best model.
- `datasets/`: Contains the METR-LA traffic dataset.
- `report/`: Contains the LaTeX progress report and generated performance plots.

## Setup Instructions

### 1. Install Dependencies
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
The METR-LA dataset is included in this repository as a compressed zip file to bypass GitHub size limits. You must unzip it before running the code:
```bash
cd datasets
unzip metr-la.h5.zip
cd ..
```

### 3. Run Training
To run the best performing Co-Fine-Tuned model:
```bash
python -m src.model5_co_finetune
```

## Architecture Evolution
This repository includes the code to replicate the 5 models discussed in our report:
1. **Model 1 (Vanilla TTM)**: Zero-shot temporal baseline (`src/model1_vanilla_ttm.py`)
2. **Model 2 (Linear Probing)**: Temporal head fine-tuning (`src/model2_linear_probing.py`)
3. **Model 3 (TTM + GCN Direct)**: Spatial adapter with direct MLP fusion (`src/model3_and_4_gcn_adapter.py`)
4. **Model 4 (TTM + GCN Residual)**: Spatial adapter learning residual corrections (`src/model3_and_4_gcn_adapter.py --use-residual`)
5. **Model 5 (Co-Fine-Tuned)**: Jointly trained temporal head and spatial adapter (`src/model5_co_finetune.py`)
