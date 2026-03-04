# Dataset Preprocessing and Evaluation Pipeline

This document explains the complete pipeline for preprocessing network traffic datasets and evaluating the machine learning models (specifically MLP) against them in the EvasionShield-LUCID framework.

## 1. Dataset Preprocessing Pipeline

The preprocessing stage transforms raw network traffic (PCAP files) or manipulated traffic into a format suitable for the machine learning models. This is primarily handled by the `process_all_folders.py` script which orchestrates `lucid_dataset_parser.py`.

### A. Input Data Structure
The system expects datasets to be organized in folders, where each folder contains:
- `*.pcap` files: Raw network traffic captures.
- `*.csv` files: Metadata or labels (optional but common).

### B. Two-Stage Processing

The preprocessing is executed in two distinct stages for each dataset folder:

#### Stage 1: Dataset Parsing (`process_all_folders.py`)
This stage reads the PCAP files and extracts flows based on 5-tuple (Source IP, Source Port, Dest IP, Dest Port, Protocol).

**Command Executed:**
```bash
python3 flatten_lucid/lucid_dataset_parser.py \
  --dataset_type DOS2019 \
  --dataset_folder <folder_path> \
  --packets_per_flow 100 \
  --dataset_id DOS2019 \
  --traffic_type all \
  --time_window 10
```

**Key Parameters:**
- `--packets_per_flow 100`: Captures the first 100 packets of each flow.
- `--time_window 10`: Sets the flow timeout window to 10 seconds.
- `--traffic_type all`: Processes all traffic types (benign and malicious).

**Output:**
- Generates intermediate `.hdf5` files containing structured flow data (Time, Length, Flags, etc.) organized in matrices (e.g., 100 packets x feature_dim).

#### Stage 2: Feature Flattening and Preprocessing
This stage takes the intermediate HDF5 files and prepares them for the MLP model. Crucially, it flattens the 2D flow matrices into 1D feature vectors because the MLP model (`10t-100n-DOS2019-LUCID-FLATTEN.h5`) expects flattened input.

**Command Executed:**
```bash
python3 flatten_lucid/lucid_dataset_parser.py \
  --preprocess_folder <folder_path> \
  --flatten \
  --dont_normalize
```

**Key Operations:**
- **Flattening**: Converts the `(100, N)` matrix of a flow into a single `(1, M)` vector of aggregated statistics (e.g., total packets, mean packet size, packet rate) or concatenated features, depending on the implementation in `lucid_dataset_parser.py`.
- **Normalization (Skipped)**: The `--dont_normalize` flag is often used here if the normalization parameters (from the training set) will be applied later during inference to ensure consistency.
- **Train/Test Split**: Although the script can split data, for evaluation purposes, we often treat the entire folder as a "Test" set if it contains specific attack scenarios.

**Final Output:**
- `*-dataset-test.hdf5`: The final preprocessed dataset ready for the model.

---

## 2. Evaluation Pipeline (Testing and Analysis)

Once the datasets are preprocessed into HDF5 format, the `test_and_analyze.py` script takes over to evaluate the model's performance.

### A. The Unified Tester
The `UnifiedTester` class is the core component. It performs the following steps:

1.  **Dataset Discovery**: Scans the `DATASETS` directory (e.g., `TrafficManipulator-master/DATASETS`) for valid folders containing `*test.hdf5` files.
2.  **Model Loading**: Loads the pre-trained MLP model (`output/10t-100n-DOS2019-LUCID-FLATTEN.h5`).

### B. Testing Loop
For each discovered dataset (e.g., `FLATTEN-PCAPS`, `MANIPULATED`, `FRAGMENTED`):

1.  **Data Loading**: Reads the `X` (features) and `y` (labels) from the HDF5 file.
2.  **Inference**:
    - Runs the model on the features: `y_pred = model.predict(X)`.
    - Applies a threshold (typically 0.5) to get binary classifications: `y_pred_bin = (y_pred > 0.5)`.
3.  **Metric Calculation**:
    - **Accuracy**: Overall correctness.
    - **F1 Score**: Harmonic mean of Precision and Recall.
    - **False Negative Rate (FNR)**: The miss rate (Crucial for verifying evasion success).
    - **False Positive Rate (FPR)**: The false alarm rate.
    - **Confusion Matrix**: Breakdowns of TP, TN, FP, FN.

### C. Analysis and Reporting
The script generates comprehensive reports stored in `ResultsFlattenLucid/run_<timestamp>_<datasets>/`:

1.  **CSV Reports**:
    - `prediction_results_*.csv`: Detailed metrics for every file/attack processed.
    - `attack_ranking_*.csv`: Attacks sorted by difficulty (FNR).
    - `summary_statistics_*.csv`: Averaged performance across datasets.

2.  **Visualizations**:
    - **FNR Comparison Plots**: Bar charts comparing the False Negative Rate of the Baseline vs. Manipulated vs. Fragmented traffic. This visually demonstrates the attack effectiveness.
    - **F1 Score Heatmaps**: Visualizes performance across different attack classes.

### D. Workflow Summary

1.  **Raw Data** (`.pcap`) $\xrightarrow{\text{process\_all\_folders.py}}$ **Intermediate HDF5**
2.  **Intermediate HDF5** $\xrightarrow{\text{preprocess + flatten}}$ **Final Test HDF5**
3.  **Final Test HDF5** + **Model** $\xrightarrow{\text{test\_and\_analyze.py}}$ **Predictions**
4.  **Predictions** $\rightarrow$ **Metrics (FNR, Acc)** $\rightarrow$ **Plots & Tables**

This pipeline ensures a consistent, reproducible method for evaluating how well the NIDS detects original attacks versus their adversarial (manipulated and fragmented) variants.

---

## 3. Extension to Other Models (Random Forest & LUCID CNN)

While the pipeline above focuses on the MLP model with flattened data, the framework also supports evaluation against other architectures: **Random Forest** and the original **LUCID CNN**.

### A. Random Forest Evaluation
The Random Forest model is another lightweight classifier used for comparison. It typically operates on the same flattened feature set as the MLP.

- **Script Location**: `RandomForest/random_forest_binary.py` or `RandomForest/rf_unified.py`
- **Data Input**: Expects the same `*-dataset-test.hdf5` files produced by the flattening preprocessing step.
- **Process**:
    1.  Loads the pre-trained Random Forest model (e.g., from `RandomForest/models/`).
    2.  Reads the flattened HDF5 datasets.
    3.  Performs inference (`model.predict()`).
    4.  Outputs similar metrics (Accuracy, F1, FNR) to compare against the MLP.

### B. LUCID CNN Evaluation
The original LUCID model is a Convolutional Neural Network (CNN) that operates on 2D matrix representations of flows (e.g., 10x10 or 100x1 matrices), rather than flattened vectors.

- **Script Location**: `lucid-ddos-master/lucid_cnn.py` or `multiclass-lucid/lucid_cnn.py`
- **Data Input**: Requires HDF5 files containing the **unflattened** matrices (i.e., skipping the `--flatten` argument during preprocessing, or using the intermediate output of `lucid_dataset_parser.py`).
- **Process**:
    1.  **Preprocessing**: Ensure `lucid_dataset_parser.py` is run *without* `--flatten`.
    2.  **Testing Command**: 
        ```bash
        python3 lucid-ddos-master/lucid_cnn.py --predict <dataset_folder> --model <path_to_cnn_model>
        ```
    3.  **Analysis**: The script calculates metrics specifically tailored for the CNN's performance, allowing a cross-architecture comparison of robustness against fragmentation and manipulation.

---
