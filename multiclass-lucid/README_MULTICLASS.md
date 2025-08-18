# LUCID Multiclass DDoS Detection

This folder contains the modified LUCID system for multiclass DDoS attack detection instead of binary classification.

## Changes Made

The following files have been modified to support multiclass classification:

### 1. `util_functions.py`
- Added `ATTACK_CLASSES` dictionary mapping attack types to class labels (0-13)
- Added `NUM_CLASSES` constant (14 classes total)
- Updated `count_packets_in_dataset()` function for better compatibility

### 2. `lucid_dataset_parser.py` 
- Added `get_attack_type_from_folder()` function to extract attack type from folder names
- Modified `parse_labels()` function to support folder-based attack type detection
- Updated `count_flows()` and `balance_dataset()` functions for multiclass support
- Enhanced dataset balancing to work across multiple attack classes

### 3. `lucid_cnn.py`
- Updated neural network architecture to use softmax activation with 14 output neurons
- Changed loss function from binary_crossentropy to categorical_crossentropy  
- Added categorical label conversion using `to_categorical()`
- Modified `report_results()` function to provide multiclass metrics and confusion matrix
- Updated prediction sections for both offline and live inference
- Enhanced evaluation with classification reports for all classes

### 4. `change_labels.py` (NEW)
- Script to convert existing HDF5 datasets from binary to multiclass labels
- Supports automatic attack type detection from filenames/folders
- Can merge multiple attack-type datasets into combined multiclass datasets
- Maintains folder structure and provides detailed statistics

## Attack Class Mapping

The system supports 14 classes (0-13):

```python
ATTACK_CLASSES = {
    'benign': 0,      
    'WebDDoS': 1,     
    'LDAP': 2,        
    'Portmap': 3,     
    'DNS': 4,         
    'UDPLag': 5,       
    'NTP': 6,         
    'SNMP': 7,         
    'SSDP': 8,        
    'Syn': 9,         
    'TFTP': 10,       
    'UDP': 11,        
    'NetBIOS': 12,    
    'MSSQL': 13       
}
```

## Usage Examples

### 1. Convert Binary Datasets to Multiclass

```bash
# Convert a single dataset folder
python3 change_labels.py /path/to/binary/datasets --output_folder /path/to/multiclass/datasets

# The script will automatically detect attack types from folder/file names
```

### 2. Merge Multiple Attack-Type Datasets  

```bash
# Merge datasets from different attack folders
python3 change_labels.py dummy \
  --merge_folders /path/to/WebDDoS /path/to/LDAP /path/to/DNS \
  --merge_output /path/to/merged/multiclass
```

### 3. Parse Network Traffic with Multiclass Labels

```bash
# Parse PCAP files with specific attack type detection
python3 lucid_dataset_parser.py \
  --dataset_type DOS2019 \
  --dataset_folder ./attack-pcaps/ \
  --packets_per_flow 10 \
  --dataset_id WebDDoS-2019 \
  --time_window 10
```

### 4. Train Multiclass Model

```bash
# Train the multiclass CNN model
python3 lucid_cnn.py \
  --train /path/to/multiclass/datasets \
  --epochs 100 \
  -cv 5
```

### 5. Test Multiclass Model

```bash  
# Test the trained multiclass model
python3 lucid_cnn.py \
  --predict /path/to/test/datasets \
  --model /path/to/trained/model.h5 \
  --iterations 5
```

### 6. Live Multiclass Detection

```bash
# Real-time multiclass detection from network interface
python3 lucid_cnn.py \
  --predict_live eth0 \
  --model /path/to/trained/model.h5 \
  --dataset_type DOS2019
```

## Dataset Organization

For optimal attack type detection, organize your datasets like this:

```
datasets/
├── 00-WebDDoS/
│   ├── train.hdf5
│   ├── val.hdf5  
│   └── test.hdf5
├── 01-LDAP/
│   ├── train.hdf5
│   ├── val.hdf5
│   └── test.hdf5
├── benign/
│   ├── train.hdf5
│   ├── val.hdf5
│   └── test.hdf5
...
```

## Model Output

The multiclass model now provides:

- **Softmax probabilities** for all 14 classes
- **Weighted F1-score** across all classes  
- **Detailed confusion matrix** showing per-class performance
- **Classification report** with precision/recall for each attack type
- **Attack percentage** instead of binary DDoS rate

## Compatibility Notes

- Existing binary models cannot be used directly with multiclass data
- Models need to be retrained on multiclass datasets
- The system maintains backward compatibility with binary datasets if needed
- All metrics are now multiclass-aware (weighted averages)

## Performance Considerations

- Slightly increased memory usage due to 14 output neurons vs 1
- Training may take longer due to increased complexity
- Inference time remains similar
- More detailed evaluation metrics available

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install tensorflow scikit-learn h5py numpy
   ```

2. **Label Conversion Issues**: Check that folder/file names contain recognizable attack type patterns

3. **Memory Issues**: Reduce batch size in training/prediction if encountering OOM errors

4. **Model Loading**: Ensure models are trained with the same NUM_CLASSES (14) configuration
