# Feature Reset Evaluation Report

**Generated**: 2026-02-28 20:16:03

**Model**: /home/rising/EvasionShield-LUCID/MLP/mlp_flatten_binary_classification.h5

**Test Data**: /home/rising/EvasionShield-LUCID/DATASETS/merged_flatten_train_dataset.hdf5

## Baseline Performance

- **Accuracy**: 0.9229
- **F1-Score**: 0.9227
- **Precision**: 0.9273
- **Recall**: 0.9229

## Model Information

- **Model Type**: Flatten
- **Input Shape**: (2650752, 21)
- **Number of Test Samples**: 2650752

## Individual Feature Analysis

- **Total Features Analyzed**: 21
- **Mean Importance Score**: 0.0585
- **Std Importance Score**: 0.1351
- **Most Important Feature**: Statistical_Feature_20 (Score: 0.4826)

### Top 10 Most Important Features

| Rank | Feature | Accuracy Drop | F1 Drop | Importance Score |
|------|---------|---------------|---------|------------------|
| 21 | Statistical_Feature_20 | 0.4123 | 0.5528 | 0.4826 |
| 7 | Statistical_Feature_6 | 0.3340 | 0.3864 | 0.3602 |
| 19 | Statistical_Feature_18 | 0.1870 | 0.1922 | 0.1896 |
| 2 | Statistical_Feature_1 | 0.1765 | 0.1818 | 0.1792 |
| 9 | Statistical_Feature_8 | 0.0527 | 0.0540 | 0.0533 |
| 18 | Statistical_Feature_17 | 0.0214 | 0.0218 | 0.0216 |
| 10 | Statistical_Feature_9 | 0.0141 | 0.0143 | 0.0142 |
| 15 | Statistical_Feature_14 | 0.0111 | 0.0112 | 0.0111 |
| 8 | Statistical_Feature_7 | 0.0064 | 0.0064 | 0.0064 |
| 20 | Statistical_Feature_19 | 0.0000 | 0.0000 | 0.0000 |

## Feature Group Analysis

| Group | Description | Accuracy Drop | F1 Drop | Importance Score |
|-------|-------------|---------------|---------|------------------|
| fourth_quarter | Fourth quarter of features | 0.4227 | 0.5684 | 0.4956 |
| second_quarter | Second quarter of features | 0.4495 | 0.4764 | 0.4629 |
| first_quarter | First quarter of features | 0.0624 | 0.0647 | 0.0635 |
| third_quarter | Third quarter of features | -0.0466 | -0.0468 | -0.0467 |

## Progressive Removal Analysis

- **Accuracy drops by 5% after removing**: 1 features
- **F1-Score drops by 5% after removing**: 1 features
- **Final performance after removing top 10 features**:
  - Accuracy: 0.5055 (-45.2%)
  - F1-Score: 0.3568 (-61.3%)

## Recommendations

Based on the feature reset evaluation:

1. **Critical Features**: 2 features have high impact (>2 standard deviations above mean)
2. **Potential for Feature Reduction**: 0 features have low impact and could potentially be removed
3. **Model Robustness**: Evaluate the model's sensitivity to feature perturbations
4. **Feature Engineering**: Consider focusing on the most important features for future improvements
5. **Adversarial Robustness**: High-impact features may be targets for adversarial attacks

## Files Generated

- `individual_feature_importance.csv`: Detailed individual feature analysis
- `feature_group_importance.csv`: Feature group analysis
- `progressive_removal.csv`: Progressive feature removal results
- `individual_feature_importance.png`: Individual feature visualizations
- `feature_group_importance.png`: Group importance visualizations
- `progressive_removal.png`: Progressive removal plots
- `feature_importance_summary.png`: Summary visualization
