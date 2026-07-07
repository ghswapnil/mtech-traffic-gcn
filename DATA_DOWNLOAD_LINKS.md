# Traffic Forecasting Datasets

Due to GitHub's strict 100MB file size limit, large spatio-temporal traffic datasets like `pems-bay.h5` (130MB) cannot be hosted directly in this repository. 

To replicate the experiments in this thesis, please download the standard pre-processed `.h5` benchmark datasets from the official academic sources listed below and place them in the `datasets/` directory.

### 1. METR-LA (Los Angeles)
- **Description:** Traffic speed readings from 207 loop detectors on the highway system of Los Angeles County over 4 months (March–June 2012).
- **Original Source:** Released by the authors of DCRNN (Diffusion Convolutional Recurrent Neural Network).
- **Download Link:** [Download METR-LA from the official DCRNN Repository](https://github.com/liyaguang/DCRNN) (Check the `data/` folder instructions in their README).

### 2. PEMS-BAY (San Francisco Bay Area)
- **Description:** Traffic speed data from 325 sensors in the San Francisco Bay Area over 6 months in 2017. 
- **Original Source:** Released by the authors of DCRNN.
- **Download Link:** [Download PEMS-BAY from the official DCRNN Repository](https://github.com/liyaguang/DCRNN) (Check the `data/` folder instructions in their README).

### 3. Alternative Direct Download (LibCity)
If the Google Drive links in the DCRNN repository are expired, you can download completely standardized versions of these datasets directly from the open-source **BigSCity-LibCity** traffic prediction library:
- **Link:** [LibCity Open Data Repository](https://github.com/LibCity/Bigscity-LibCity-Datasets)

---

### Setup Instructions
Once downloaded, ensure your `datasets/` directory looks like this before running the training scripts:

```text
datasets/
├── metr-la.h5
├── pems-bay.h5
└── pems-bay-meta.h5
```
