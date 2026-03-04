# Comprehensive Evaluation Pipeline Analysis

## Abstract

This document details the complete experimental pipeline developed to evaluate the robustness of machine learning-based Network Intrusion Detection Systems (NIDS) against adversarial traffic. The pipeline integrates data preprocessing, feature extraction, and a comparative analysis of three distinct detection architectures: LUCID (a lightweight Convolutional Neural Network), a Multi-Layer Perceptron (MLP), and a Random Forest (RF) classifier. The evaluation assesses the efficacy of adversarial evasion techniques—specifically GAN-PSO manipulation and Stochastic IP Fragmentation—across these models.

## 1. Introduction

To rigorously assess the security of ML-NIDS, a standardized evaluation framework is required. This framework must handle the ingestion of raw network traffic (PCAP), transform it into model-specific feature representations, train baseline detectors, and quantify their performance degradation when exposed to adversarial samples.

The pipeline comprises three main stages:
1.  **Preprocessing**: Converting raw PCAP files into structural features suitable for deep learning and tree-based models.
2.  **Model Training**: Establishing baseline performance on clean traffic for LUCID, MLP, and RF.
3.  **Adversarial Evaluation**: Testing these trained models against manipulated and fragmented traffic variants to measure evasion success rates.

## 2. Preprocessing Pipeline

The preprocessing stage transforms raw packet captures into a unified feature set. This process is critical for ensuring that all models are evaluated on identical data representations.

### 2.1 Packet Feature Extraction (`lucid_dataset_parser.py`)

The extraction logic converts raw PCAP files into numerical feature vectors. For every packet, an 11-dimensional feature vector is extracted:

| Index | Feature Description | Data Type | Notes |
| :--- | :--- | :--- | :--- |
| 0 | Packet Timestamp | Float | Normalized inter-arrival time |
| 1 | Packet Length | Integer | Frame length in bytes |
| 2 | Highest Layer | Integer | Protocol hierarchy hash |
| 3 | IP Flags | Integer | Bitmask of IP flags |
| 4 | Protocols | Integer | One-hot encoded protocol stack |
| 5 | TCP Length | Integer | Payload size of TCP segment |
| 6 | TCP ACK | Integer | Acknowledgment number |
| 7 | TCP Flags | Integer | Bitmask of TCP control flags |
| 8 | TCP Window Size | Integer | Flow control window |
| 9 | UDP Length | Integer | Payload size of UDP datagram |
| 10 | ICMP Type | Integer | Control message type |

**Code Implementation:**
The `parse_packet` function leverages `pyshark` to decode protocol fields:

```python
def parse_packet(pkt):
    pf = packet_features()
    # ... timestamp and IP fields ...
    pf.features_list.append(float(pkt.sniff_timestamp))
    pf.features_list.append(int(getattr(pkt.ip, 'len', 0)))
    
    # ... protocol specific extraction ...
    if protocol == socket.IPPROTO_TCP:
        pf.features_list.append(int(getattr(pkt.tcp, 'len', 0)))
        pf.features_list.append(int(pkt.tcp.ack))
        pf.features_list.append(int(pkt.tcp.flags, 16))
```

### 2.2 Dataset Splitting and Normalization

The extracted flows are aggregated into HDF5 datasets. To ensure rigorous evaluation, the data is split into disjoint sets:
*   **Training Set**: Used to optimize model weights (Clean traffic only).
*   **Validation Set**: Used for hyperparameter tuning and early stopping.
*   **Testing Set**: Used for final evaluation (Contains Clean, Manipulated, and Fragmented variants).

The features are normalized to the range $[0, 1]$ to facilitate neural network convergence.

## 3. Detection Models

Three distinct architectures are implemented to represent a broad spectrum of NIDS complexity.

### 3.1 LUCID: Lightweight CNN (`lucid_cnn_flatten.py`)
LUCID is a 1D Convolutional Neural Network designed for resource-constrained environments. It processes traffic as a time-series of packet features.

*   **Input Shape**: $(T, F) = (10, 11)$ representing the first 10 packets of a flow with 11 features each.
*   **Architecture**:
    *   Conv1D Layer (Filters=64, Kernel=3, Activation=ReLU)
    *   Max Pooling
    *   Flatten
    *   Dense Layer (Units=32, Activation=ReLU)
    *   Dropout (Rate=0.5)
    *   Output Layer (Sigmoid for Binary Classification)

### 3.2 Multi-Layer Perceptron (MLP)
A standard feed-forward neural network serving as a baseline for deep learning performance without spatial/temporal inductive biases.

*   **Input**: Flattened vector of size $10 \times 11 = 110$.
*   **Architecture**:
    *   Dense Block 1: 128 Units + ReLU + BatchNormalization
    *   Dense Block 2: 64 Units + ReLU + Dropout
    *   Output: Sigmoid

### 3.3 Random Forest (`rf_unified.py`)
An ensemble learning method representing traditional machine learning approaches. It is generally robust to noise but lacks the capacity to learn complex temporal sequences.

*   **Configuration**:
    *   Estimators: 100
    *   Criterion: Gini Impurity
    *   Max Depth: None (Fully grown trees)

## 4. Evaluation Methodology

The core of the study is the comparative analysis of these models across three distinct datasets representing progressive levels of adversarial sophistication.

### 4.1 Datasets

1.  **Baseline (FLATTEN-PCAPS)**:
    *   Original, unmodified traffic from the CIC-IDS2019 dataset.
    *   Represents the "best-case" scenario for the defender.

2.  **Manipulated (MANIPULATED-FLATTEN_42)**:
    *   Traffic modified by the GAN-PSO Traffic Manipulator.
    *   PERTURBATIONS: Adjusted inter-arrival times and dummy payload injection.
    *   Goal: Test robustness against feature-space optimization.

3.  **Fragmented (FRAGMENTED_42_150)**:
    *   Manipulation augmented with the **Stochastic Fragmentation Module**.
    *   PERTURBATIONS: Probabilistic packet decomposition ($\rho=1.0$, $S_{max}=150$).
    *   Goal: Test robustness against structural modification.

### 4.2 Metrics

The primary metric for adversarial success is the **False Negative Rate (FNR)**, defined as:

$$
FNR = \frac{FN}{FN + TP}
$$

An increase in FNR from Baseline $\to$ Manipulated $\to$ Fragmented indicates successful evasion.

### 4.3 Unified Analysis Pipeline

The evaluation is orchestrated via a unified script that loads the pre-trained models and iterates through the test sets.

```python
# Pseudo-code for evaluation loop
results = {}
for model in [LUCID, MLP, RF]:
    for dataset in [BASE, MANIP, FRAG]:
        X_test, y_test = load_dataset(dataset)
        preds = model.predict(X_test)
        
        # Calculate Attack-Specific FNR
        for attack_type in attacks:
            idx = indices[attack_type]
            fnr = 1 - recall_score(y_test[idx], preds[idx])
            results[model][dataset][attack_type] = fnr
```

## 5. Summary of Experimental Flow

1.  **Ingestion**: Raw packets are parsed into 11-feature vectors.
2.  **Training**: LUCID, MLP, and RF are trained on *clean* traffic only, simulating a defender who is unaware of the specific attack strategy.
3.  **Attack Generation**:
    *   Base traffic is perturbed by GAN-PSO.
    *   Resulting traffic is processed by the Fragmentation Module.
4.  **Testing**: The models classify the generated adversarial samples.
5.  **Comparison**: The degradation in detection rate (FNR increase) quantifies the "Adversarial Advantage."

This rigorous pipeline ensures that any observed drop in performance is strictly attributable to the adversarial transformations, providing a fair and reproducible assessment of NIDS vulnerability.
