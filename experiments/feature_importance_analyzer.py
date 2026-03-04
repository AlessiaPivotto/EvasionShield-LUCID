#!/usr/bin/env python3
"""
Feature Importance Analysis for LUCID DDoS Detection
Identifies the most important features in your specific datasets
"""

import os
import sys
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_h5_dataset(filepath, max_samples=2000):
    """Load data from HDF5 file"""
    print(f"Loading: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Get features and labels using the correct keys
        key_combinations = [
            ('set_x', 'set_y'),
            ('X', 'y'),
            ('x', 'y'),
            ('data', 'labels'),
        ]
        
        X, y = None, None
        for x_key, y_key in key_combinations:
            if x_key in f and y_key in f:
                X = f[x_key][:]
                y = f[y_key][:]
                break
        
        if X is None:
            raise ValueError(f"Could not find data in {filepath}")
        
        # Limit samples
        if max_samples and X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X, y

def analyze_feature_importance(X, y, dataset_name):
    """Analyze feature importance using Random Forest"""
    print(f"\n=== Feature Importance Analysis for {dataset_name.upper()} ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get baseline accuracy
    baseline_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Get feature importance scores
    feature_importance = rf.feature_importances_
    
    # Create feature ranking
    feature_ranking = [(i, importance) for i, importance in enumerate(feature_importance)]
    feature_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFeature Importance Ranking (Top 10):")
    print("-" * 50)
    for rank, (feature_idx, importance) in enumerate(feature_ranking[:10], 1):
        print(f"{rank:2d}. Feature {feature_idx:2d}: {importance:.4f}")
    
    return feature_importance, feature_ranking, rf, (X_test, y_test)

def test_individual_features(rf, X_test, y_test, feature_ranking, dataset_name, top_n=10):
    """Test impact of removing individual features"""
    print(f"\n=== Individual Feature Impact Analysis for {dataset_name.upper()} ===")
    
    baseline_acc = accuracy_score(y_test, rf.predict(X_test))
    
    feature_impacts = []
    
    print("\nTesting individual feature removal (Top 10 most important):")
    print("-" * 70)
    
    for rank, (feature_idx, importance) in enumerate(feature_ranking[:top_n], 1):
        # Create test data with this feature zeroed
        X_test_modified = X_test.copy()
        X_test_modified[:, feature_idx] = 0
        
        # Test accuracy
        modified_acc = accuracy_score(y_test, rf.predict(X_test_modified))
        impact = baseline_acc - modified_acc
        relative_impact = (impact / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        feature_impacts.append({
            'feature_idx': feature_idx,
            'importance_score': importance,
            'accuracy_drop': impact,
            'relative_drop': relative_impact,
            'rank': rank
        })
        
        print(f"{rank:2d}. Feature {feature_idx:2d} | Importance: {importance:.4f} | "
              f"Drop: {impact:.4f} ({relative_impact:5.1f}%)")
    
    return feature_impacts

def analyze_feature_combinations(rf, X_test, y_test, feature_ranking, dataset_name):
    """Test impact of removing combinations of top features"""
    print(f"\n=== Feature Combination Analysis for {dataset_name.upper()} ===")
    
    baseline_acc = accuracy_score(y_test, rf.predict(X_test))
    
    # Test removing top N features together
    top_features = [idx for idx, _ in feature_ranking[:5]]  # Top 5 features
    
    combination_results = []
    
    for n in range(1, 6):  # Test removing 1 to 5 top features
        features_to_remove = top_features[:n]
        
        X_test_modified = X_test.copy()
        X_test_modified[:, features_to_remove] = 0
        
        modified_acc = accuracy_score(y_test, rf.predict(X_test_modified))
        impact = baseline_acc - modified_acc
        relative_impact = (impact / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        combination_results.append({
            'num_features_removed': n,
            'features_removed': features_to_remove,
            'accuracy_drop': impact,
            'relative_drop': relative_impact
        })
        
        print(f"Removing top {n} feature{'s' if n > 1 else ''}: {features_to_remove}")
        print(f"  Accuracy drop: {impact:.4f} ({relative_impact:5.1f}%)")
        print()
    
    return combination_results

def create_lucid_feature_map():
    """Create mapping of feature indices to LUCID network flow features"""
    # Based on LUCID's feature extraction (this is an educated guess based on common network flow features)
    feature_map = {
        0: "Flow Duration",
        1: "Total Forward Packets", 
        2: "Total Backward Packets",
        3: "Total Length Forward Packets",
        4: "Total Length Backward Packets",
        5: "Forward Packet Length Max",
        6: "Forward Packet Length Min", 
        7: "Forward Packet Length Mean",
        8: "Forward Packet Length Std",
        9: "Backward Packet Length Max",
        10: "Backward Packet Length Min",
        11: "Backward Packet Length Mean", 
        12: "Backward Packet Length Std",
        13: "Flow Bytes/s",
        14: "Flow Packets/s",
        15: "Flow IAT Mean",
        16: "Flow IAT Std",
        17: "Forward IAT Total",
        18: "Forward IAT Mean",
        19: "Forward IAT Std", 
        20: "Backward IAT Total"
    }
    return feature_map

def analyze_all_datasets():
    """Analyze feature importance across all datasets"""
    
    datasets = {
        'baseline': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'manipulated': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42/09-TFTP/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'fragmented': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FRAGMENTED_42_150/09-TFTP/10t-100n-DOS2019-flatten-dataset-test.hdf5'
    }
    
    feature_map = create_lucid_feature_map()
    all_results = {}
    
    print("=" * 80)
    print("LUCID FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue
            
        try:
            # Load data
            X, y = load_h5_dataset(dataset_path)
            
            # Analyze feature importance
            importance_scores, feature_ranking, model, test_data = analyze_feature_importance(X, y, dataset_name)
            
            # Test individual features
            individual_impacts = test_individual_features(model, test_data[0], test_data[1], 
                                                        feature_ranking, dataset_name)
            
            # Test feature combinations
            combination_impacts = analyze_feature_combinations(model, test_data[0], test_data[1], 
                                                             feature_ranking, dataset_name)
            
            all_results[dataset_name] = {
                'importance_scores': importance_scores,
                'feature_ranking': feature_ranking,
                'individual_impacts': individual_impacts,
                'combination_impacts': combination_impacts
            }
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Create summary analysis
    print("\n" + "=" * 80)
    print("CROSS-DATASET FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)
    
    # Find most critical features across all datasets
    critical_features = {}
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} - Most Critical Features:")
        print("-" * 50)
        
        for impact_data in results['individual_impacts'][:5]:
            feature_idx = impact_data['feature_idx']
            feature_name = feature_map.get(feature_idx, f"Feature {feature_idx}")
            
            if feature_idx not in critical_features:
                critical_features[feature_idx] = []
            critical_features[feature_idx].append({
                'dataset': dataset_name,
                'impact': impact_data['relative_drop'],
                'rank': impact_data['rank']
            })
            
            print(f"  {impact_data['rank']:2d}. {feature_name:<25} | "
                  f"Impact: {impact_data['relative_drop']:5.1f}%")
    
    # Find features that are critical across multiple datasets
    print(f"\n" + "=" * 80)
    print("FEATURES CRITICAL ACROSS MULTIPLE DATASETS")
    print("=" * 80)
    
    multi_dataset_critical = {}
    for feature_idx, dataset_impacts in critical_features.items():
        if len(dataset_impacts) >= 2:  # Critical in 2+ datasets
            avg_impact = np.mean([d['impact'] for d in dataset_impacts])
            multi_dataset_critical[feature_idx] = {
                'avg_impact': avg_impact,
                'datasets': dataset_impacts,
                'feature_name': feature_map.get(feature_idx, f"Feature {feature_idx}")
            }
    
    # Sort by average impact
    sorted_critical = sorted(multi_dataset_critical.items(), 
                           key=lambda x: x[1]['avg_impact'], reverse=True)
    
    print("\nMost universally critical features:")
    print("-" * 60)
    for feature_idx, data in sorted_critical:
        print(f"Feature {feature_idx:2d}: {data['feature_name']:<25} | Avg Impact: {data['avg_impact']:5.1f}%")
        for dataset_info in data['datasets']:
            print(f"    {dataset_info['dataset']:>12}: {dataset_info['impact']:5.1f}% (rank {dataset_info['rank']})")
        print()
    
    # Save detailed results
    save_detailed_results(all_results, feature_map)
    
    return all_results

def save_detailed_results(all_results, feature_map):
    """Save detailed feature importance results"""
    
    # Create comprehensive results dataframe
    all_individual_impacts = []
    
    for dataset_name, results in all_results.items():
        for impact_data in results['individual_impacts']:
            feature_idx = impact_data['feature_idx']
            all_individual_impacts.append({
                'dataset': dataset_name,
                'feature_idx': feature_idx,
                'feature_name': feature_map.get(feature_idx, f"Feature {feature_idx}"),
                'importance_score': impact_data['importance_score'],
                'accuracy_drop': impact_data['accuracy_drop'],
                'relative_drop': impact_data['relative_drop'],
                'rank': impact_data['rank']
            })
    
    df = pd.DataFrame(all_individual_impacts)
    
    # Save CSV
    csv_path = '/home/rising/EvasionShield-LUCID/experiments/feature_importance_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature importance heatmap
    plt.subplot(2, 2, 1)
    pivot_importance = df.pivot(index='feature_name', columns='dataset', values='relative_drop')
    sns.heatmap(pivot_importance, annot=True, fmt='.1f', cmap='Reds', cbar=True)
    plt.title('Feature Impact Across Datasets (%)')
    plt.ylabel('Features')
    plt.xticks(rotation=45)
    
    # Plot 2: Top features by dataset
    plt.subplot(2, 2, 2)
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset].head(5)
        plt.bar(range(len(dataset_data)), dataset_data['relative_drop'], 
               label=dataset, alpha=0.7)
    plt.xlabel('Feature Rank')
    plt.ylabel('Relative Impact (%)')
    plt.title('Top 5 Features by Dataset')
    plt.legend()
    
    # Plot 3: Feature importance scores
    plt.subplot(2, 2, 3)
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset].head(5)
        plt.plot(range(1, 6), dataset_data['importance_score'], 
                marker='o', label=dataset, linewidth=2)
    plt.xlabel('Feature Rank')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Average impact across datasets
    plt.subplot(2, 2, 4)
    avg_impact = df.groupby('feature_name')['relative_drop'].mean().sort_values(ascending=False).head(10)
    plt.barh(range(len(avg_impact)), avg_impact.values)
    plt.yticks(range(len(avg_impact)), avg_impact.index)
    plt.xlabel('Average Impact (%)')
    plt.title('Most Critical Features (Average)')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/home/rising/EvasionShield-LUCID/experiments/feature_importance_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Starting comprehensive feature importance analysis...")
    results = analyze_all_datasets()
