# Random Forest Binary Classification for DDoS Detection

This directory contains a Random Forest implementation for binary classification of DDoS attacks that is compatible with Flatten LUCID input format.

## Overview

The Random Forest classifier is designed to:
- Take the same flattened input as Flatten LUCID (21 statistical features extracted from network flows)
- Perform binary classification (Benign vs DDoS)
- Provide comprehensive evaluation metrics and visualization capabilities
- Support hyperparameter tuning and cross-validation

## Files

- `random_forest_binary.py` - Main Random Forest classifier implementation
- `rf_utils.py` - Utility functions for visualization and analysis
- `example_usage.py` - Example script demonstrating usage
- `README.md` - This documentation file

## Requirements

The implementation requires the following Python packages:
```bash
pip install numpy pandas scikit-learn h5py matplotlib seaborn
```

## Input Format

The Random Forest classifier expects the same input format as Flatten LUCID:
- **HDF5 files** containing:
  - `set_x`: Feature matrix (samples × 21 features)
  - `set_y`: Label vector (0 = Benign, 1 = DDoS)

The 21 features are statistical summaries extracted from network flows:
1. Time feature (mean time difference)
2. Packet length (mean)
3-5. IP flags (sum of don't fragment, more fragments, reserved bit)
6. Fragment offset (mean)
7. Protocols (mean)
8. TCP length (mean)
9-15. TCP flags (sum of each flag type)
16-19. UDP features
20. ICMP features
21. Packet count

## Usage

### Basic Training
```bash
python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --val ../DATASETS/merged_flatten_val_dataset.hdf5
```

### Training with Hyperparameter Tuning
```bash
python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --tune --tune_method random --n_iter 20
```

### Cross-Validation
```bash
python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --cv 5
```

### Prediction/Testing
```bash
python3 random_forest_binary.py --predict ../DATASETS/merged_flatten_test_dataset.hdf5 --model random_forest_model.pkl
```

### Show Feature Importance
```bash
python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5 --feature_importance
```

## Command Line Arguments

### Training Arguments
- `--train PATH` - Path to training dataset (HDF5 file)
- `--val PATH` - Path to validation dataset (HDF5 file) [optional]
- `--cv N` - Number of cross-validation folds [default: 0]
- `--model PATH` - Path to save/load model file [default: auto-generated]

### Hyperparameter Tuning
- `--tune` - Enable hyperparameter tuning
- `--tune_method {grid,random}` - Tuning method [default: random]
- `--n_iter N` - Number of iterations for random search [default: 20]

### Model Parameters
- `--n_estimators N` - Number of trees [default: 100]
- `--max_depth N` - Maximum tree depth [default: None]

### Prediction Arguments
- `--predict PATH` - Path to test dataset for prediction
- `--model PATH` - Path to trained model file (required for prediction)

### Analysis Arguments
- `--feature_importance` - Show feature importance after training

## Example Workflows

### 1. Quick Start with Merged Dataset
```bash
# Assuming you have run merge_hdf5.py to create merged datasets
cd RandomForest

# Train a basic model
python3 random_forest_binary.py --train ../DATASETS/merged_flatten_train_dataset.hdf5

# Test the model (create test dataset first if not available)
python3 random_forest_binary.py --predict ../DATASETS/merged_flatten_test_dataset.hdf5 --model random_forest_binary_*.pkl
```

### 2. Complete Training with Validation and Cross-Validation
```bash
# Train with validation and 5-fold cross-validation
python3 random_forest_binary.py \
    --train ../DATASETS/merged_flatten_train_dataset.hdf5 \
    --cv 5 \
    --feature_importance \
    --model my_rf_model.pkl
```

### 3. Hyperparameter Tuning
```bash
# Perform hyperparameter tuning with random search
python3 random_forest_binary.py \
    --train ../DATASETS/merged_flatten_train_dataset.hdf5 \
    --tune \
    --tune_method random \
    --n_iter 50 \
    --cv 5
```

### 4. Running Examples
```bash
# Run the example script to see various capabilities
python3 example_usage.py
```

## Output

### Training Output
- Model performance metrics (accuracy, precision, recall, F1-score)
- Cross-validation scores (if enabled)
- Feature importance rankings (if enabled)
- Saved model file (.pkl format)

### Prediction Output
- Comprehensive evaluation metrics
- Confusion matrix
- Classification report
- CSV file with predictions and probabilities

### Files Generated
- `random_forest_binary_<timestamp>.pkl` - Trained model
- `rf_predictions_<timestamp>.csv` - Prediction results

## Performance Comparison with Flatten LUCID

The Random Forest implementation provides several advantages:

### Advantages
- **Fast Training**: Much faster than neural networks
- **No GPU Required**: Runs efficiently on CPU
- **Interpretability**: Feature importance rankings
- **Robustness**: Less prone to overfitting
- **No Hyperparameter Sensitivity**: Good default performance

### Expected Performance
On typical DDoS datasets, Random Forest should achieve:
- **Accuracy**: 95-99%
- **F1-Score**: 0.95-0.99
- **Training Time**: Seconds to minutes (vs hours for deep learning)
- **Inference Time**: Milliseconds per sample

## Integration with Flatten LUCID Pipeline

The Random Forest classifier can be used as a drop-in replacement for the neural network in Flatten LUCID:

1. **Data Preprocessing**: Use the same `lucid_dataset_parser.py` with `--flatten` option
2. **Feature Format**: Same 21-feature statistical representation  
3. **Input/Output**: Compatible HDF5 format
4. **Evaluation**: Same metrics and analysis tools

## Troubleshooting

### Common Issues

1. **"No files found matching pattern"**
   - Check file paths are correct
   - Ensure HDF5 files exist and are accessible

2. **"Model must be trained before prediction"**
   - Train a model first or load an existing model file

3. **Memory errors with large datasets**
   - Use data subsampling for hyperparameter tuning
   - Process data in batches if needed

4. **Import errors**
   - Install required packages: `pip install numpy pandas scikit-learn h5py matplotlib seaborn`

### Performance Tips

1. **For large datasets**: Use `n_jobs=-1` to utilize all CPU cores
2. **For faster tuning**: Reduce `n_iter` or use subset of data
3. **For better performance**: Increase `n_estimators` (more trees)
4. **For faster inference**: Decrease `n_estimators` or `max_depth`

## Comparison with Other Models

| Model | Training Speed | Inference Speed | Interpretability | Accuracy |
|-------|---------------|-----------------|------------------|----------|
| Random Forest | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Flatten LUCID | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| SVM | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## License

This implementation follows the same Apache 2.0 license as the original Flatten LUCID project.
