# Feature Reset Evaluation for LUCID - Usage Guide

This directory contains tools for evaluating feature importance in LUCID DDoS detection models using **feature reset (zeroing) techniques**. This approach helps understand which features are most critical for detection performance.

## What is Feature Reset Evaluation?

Feature reset evaluation systematically sets features to zero and measures the impact on model performance. This helps identify:
- **Critical features** that significantly impact accuracy when removed
- **Redundant features** that have minimal impact
- **Feature robustness** and model sensitivity
- **Attack vectors** for adversarial manipulation

## Available Tools

### 1. `feature_reset_example.py` - Quick Demo
A user-friendly demo script that gets you started quickly.

```bash
# Check dependencies and scan for available files
python feature_reset_example.py --check-deps
python feature_reset_example.py --scan

# Run demo with your model and data
python feature_reset_example.py --model path/to/model.h5 --data path/to/test.hdf5
```

**Example with your files:**
```bash
# For a CNN model
python feature_reset_example.py --model ../output/10t-100n-DOS2019-LUCID.h5 --data ../sample-dataset/10t-100n-DOS2019-dataset-test.hdf5

# For a flatten/MLP model  
python feature_reset_example.py --model ../MLP/mlp_flatten_binary_classification.h5 --data ../DATASETS/merged_flatten_train_dataset.hdf5
```

### 2. `lucid_feature_analyzer.py` - Comprehensive Analysis
Advanced tool specifically designed for LUCID models with detailed packet and temporal analysis.

```bash
# Basic analysis
python lucid_feature_analyzer.py --model model.h5 --data test.hdf5

# Quick analysis (faster, reduced scope)
python lucid_feature_analyzer.py --model model.h5 --data test.hdf5 --quick

# Custom parameters
python lucid_feature_analyzer.py --model model.h5 --data test.hdf5 \
    --max-samples 500 --max-packets 20 --max-features 6 \
    --output my_analysis_results
```

### 3. `feature_reset_evaluation.py` - Full Framework
Complete feature reset evaluation framework with extensive analysis capabilities.

```bash
# Complete evaluation
python feature_reset_evaluation.py --model model.h5 --data test.hdf5

# Individual features only
python feature_reset_evaluation.py --model model.h5 --data test.hdf5 --individual_only

# Feature groups only  
python feature_reset_evaluation.py --model model.h5 --data test.hdf5 --groups_only
```

## Understanding the Results

### For CNN Models (Original LUCID)
Features are analyzed as:
- **Packet Position**: Which packets in the time window are most important
- **Feature Type**: Which packet attributes (IAT, length, flags, etc.) are critical
- **Temporal Patterns**: Early vs middle vs late packets in flows
- **Individual Combinations**: Specific packet/feature pairs

### For Flatten Models (Statistical LUCID)
Features represent statistical aggregations:
- **Statistical Features**: Mean, std, min, max of various packet attributes
- **Feature Groups**: Related statistical measures
- **Feature Importance Ranking**: Individual feature contributions

### Key Metrics
- **Accuracy Drop**: How much accuracy decreases when feature is zeroed
- **F1-Score Drop**: Impact on F1-score
- **Importance Score**: Combined metric of feature criticality
- **Relative Importance**: Percentage impact relative to baseline

## Typical Workflow

1. **Start with the demo** to get familiar:
   ```bash
   python feature_reset_example.py --scan
   python feature_reset_example.py --model <found_model> --data <found_data>
   ```

2. **Run comprehensive analysis** on your best model:
   ```bash
   python lucid_feature_analyzer.py --model your_best_model.h5 --data test_data.hdf5
   ```

3. **Analyze results** in the generated reports and visualizations

4. **Use insights** for:
   - Model optimization
   - Feature selection
   - Robustness testing
   - Adversarial defense

## Output Files

Each tool generates:
- **CSV files**: Numerical results for each analysis
- **PNG plots**: Visualizations of feature importance
- **Markdown reports**: Comprehensive analysis summaries
- **Heatmaps**: Feature importance matrices (for CNN models)

## Example Command Sequences

### Quick Start with Existing Models
```bash
# Find available models and datasets
python feature_reset_example.py --scan

# Test with found files
python feature_reset_example.py --model "./output/10t-100n-DOS2019-LUCID.h5" --data "./sample-dataset/10t-100n-DOS2019-dataset-test.hdf5"
```

### Comprehensive Analysis
```bash
# Full LUCID feature analysis
python lucid_feature_analyzer.py \
    --model "./output/10t-100n-DOS2019-LUCID.h5" \
    --data "./sample-dataset/10t-100n-DOS2019-dataset-test.hdf5" \
    --output "./analysis_results" \
    --max-samples 1000
```

### Advanced Evaluation
```bash
# Complete feature reset framework
python feature_reset_evaluation.py \
    --model "./output/10t-100n-DOS2019-LUCID.h5" \
    --data "./sample-dataset/10t-100n-DOS2019-dataset-test.hdf5" \
    --output "./complete_evaluation" \
    --max_progressive 25
```

## Dependencies

Required packages:
- tensorflow
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- h5py

Install with:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn h5py
```

## Tips for Best Results

1. **Use representative test data** that wasn't used for training
2. **Start with smaller samples** to test the process quickly
3. **Focus on your best-performing models** first
4. **Compare results across different model architectures**
5. **Use the insights to improve model robustness**

## Interpreting Results for LUCID

### High-Impact Features May Indicate:
- **Critical attack signatures** in packet timing or sizes
- **Important flow characteristics** for DDoS detection
- **Temporal patterns** specific to attack types
- **Vulnerable points** for adversarial attacks

### Low-Impact Features May Suggest:
- **Redundant information** that could be removed
- **Noise in the data** that doesn't contribute to detection
- **Over-engineering** in feature extraction
- **Opportunities for model simplification**

This analysis will help you understand your LUCID models better and make them more robust against evasion attacks!
