# Spatio-Temporal Traffic Forecasting Datasets

Due to GitHub's strict 100MB file size limit, large spatio-temporal traffic datasets cannot be hosted directly in this repository. 

Below is a comprehensive directory of the canonical benchmark datasets relevant for Graph Neural Network (GNN) traffic forecasting, along with their official academic sources and direct download links. 

*Note on Data Modalities: When selecting a dataset, be aware of the difference between fixed-sensor data and probe-vehicle data. Fixed-sensor datasets (like METR-LA and PeMS) are recorded by loop detectors buried on major freeways; while excellent for time-series forecasting, they do not provide coverage for minor city streets. Probe-vehicle datasets (like NYC Taxi trajectories) are recorded by moving GPS devices and provide true, dense spatial coverage across the entire physical urban grid.*

---

## 1. Traffic Speed Datasets
These datasets measure the average velocity of vehicles passing through highway sensors. They are the primary benchmarks used to evaluate models like STGCN, DCRNN, Graph WaveNet, and our **TTM+GCN Adapter**.

*   **METR-LA (Los Angeles)**
    *   **Description:** Speed readings from 207 loop detectors in Los Angeles County over 4 months (March–June 2012).
    *   **Download:** [DCRNN Official Google Drive](https://github.com/liyaguang/DCRNN#data-preparation) | [LibCity Mirror](https://github.com/LibCity/Bigscity-LibCity-Datasets)

*   **PEMS-BAY (San Francisco Bay Area)**
    *   **Description:** Speed data from 325 sensors in the San Francisco Bay Area over 6 months in 2017. 
    *   **Download:** [DCRNN Official Google Drive](https://github.com/liyaguang/DCRNN#data-preparation) | [LibCity Mirror](https://github.com/LibCity/Bigscity-LibCity-Datasets)

*   **Seattle Freeway Dataset**
    *   **Description:** Traffic speed data from 323 loop detectors in Seattle over a one-month period.
    *   **Download:** [Transdim Repository (Seattle Folder)](https://github.com/xinychen/transdim/tree/master/datasets/Seattle-data-set)

## 2. Traffic Flow / Volume Datasets
These datasets predict *how many* cars are passing through a sensor per 5-minute interval (traffic volume).

*   **PeMSD4 (San Francisco Bay Area)** & **PeMSD8 (San Bernardino)**
    *   **Description:** Traffic flow data from 307 and 170 sensors respectively.
    *   **Download:** [ASTGCN Official Repository](https://github.com/Davidham3/ASTGCN#data-preparation)

*   **PeMSD7 (California)**
    *   **Description:** Traffic flow data from 228 sensors.
    *   **Download:** [STGCN Official Repository](https://github.com/VeritasYin/STGCN_IJCAI-18#data-preparation)

*   **EXPY-TKY (Tokyo Expressway)**
    *   **Description:** Traffic volume data on the Tokyo highway network.
    *   **Download:** [LibCity Mirror](https://github.com/LibCity/Bigscity-LibCity-Datasets)

## 3. Large-Scale Foundation Model Datasets
These modern datasets are massively larger, designed to test the scalability of models and zero-shot foundation models (like TTM). 

*   **LargeST Benchmark**
    *   **Description:** A massive repository containing data from **8,600 sensors** covering a 5-year period in California. Specifically built to overwhelm traditional models.
    *   **Download:** [LargeST Official GitHub](https://github.com/liuxu77/LargeST)

*   **Raw Caltrans PeMS Data**
    *   **Description:** The primary, continuously updated source of all California traffic data. Researchers can extract custom time periods and regions.
    *   **Download:** [Caltrans Performance Measurement System Data Portal](https://pems.dot.ca.gov/) (Requires free registration)

## 4. Urban Mobility & Trajectory Datasets
These datasets utilize probe-vehicle GPS pings to capture true spatial coverage across dense city grids.

*   **NYC Taxi Dataset**
    *   **Description:** Records pickup/drop-off locations across Manhattan zones, used for true spatial trajectory forecasting across a dense city grid.
    *   **Download:** [NYC TLC Trip Record Data Portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

*   **T-Drive Trajectory Data**
    *   **Description:** GPS trajectories of over 10,000 taxis traversing the dense urban grid of Beijing.
    *   **Download:** [Microsoft Research T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)

---

### General Setup Instructions
To run the models in this repository, download `metr-la.h5` from the sources above and place it in the `datasets/` directory:

```text
datasets/
├── metr-la.h5
└── pems-bay.h5 (optional, for zero-shot transferability testing)
```
