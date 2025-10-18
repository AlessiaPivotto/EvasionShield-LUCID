#!/usr/bin/env python3

"""
Utility functions for Random Forest Binary Classification
Compatible with Flatten LUCID input format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import h5py
import glob

def plot_confusion_matrix(cm, classes=['Benign', 'DDoS'], title='Confusion Matrix'):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    return plt

def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_precision_recall_curve(y_true, y_prob, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_feature_importance(importance_df, top_n=20, title='Feature Importance'):
    """
    Plot feature importance
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title(f'{title} (Top {top_n})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt

def compare_models_performance(results_dict, metrics=['accuracy', 'precision', 'recall', 'f1_score']):
    """
    Compare performance of different models
    
    Args:
        results_dict: Dictionary with model names as keys and metrics dict as values
        metrics: List of metrics to compare
    """
    comparison_df = pd.DataFrame(results_dict).T[metrics]
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        comparison_df[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return plt, comparison_df

def analyze_dataset_statistics(hdf5_file):
    """
    Analyze dataset statistics
    """
    with h5py.File(hdf5_file, 'r') as f:
        X = np.array(f['set_x'][:])
        y = np.array(f['set_y'][:])
    
    # Flatten if needed
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    stats = {
        'total_samples': X.shape[0],
        'num_features': X.shape[1],
        'benign_samples': np.sum(y == 0),
        'malicious_samples': np.sum(y == 1),
        'class_balance': np.sum(y == 0) / np.sum(y == 1) if np.sum(y == 1) > 0 else np.inf,
        'feature_means': np.mean(X, axis=0),
        'feature_stds': np.std(X, axis=0),
        'feature_mins': np.min(X, axis=0),
        'feature_maxs': np.max(X, axis=0)
    }
    
    print(f"Dataset Statistics for {hdf5_file}:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Number of features: {stats['num_features']:,}")
    print(f"  Benign samples: {stats['benign_samples']:,} ({stats['benign_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Malicious samples: {stats['malicious_samples']:,} ({stats['malicious_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Class balance ratio (Benign:Malicious): {stats['class_balance']:.2f}:1")
    
    return stats

def create_model_report(model, X_test, y_test, model_name="Random Forest"):
    """
    Create a comprehensive model report
    """
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions)
    }
    
    # Create report
    report = f"""
    ====== {model_name} Model Performance Report ======
    
    Test Set Size: {len(y_test):,} samples
    
    Metrics:
    --------
    Accuracy:  {metrics['accuracy']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall:    {metrics['recall']:.4f}
    F1-Score:  {metrics['f1_score']:.4f}
    
    Detailed Classification Report:
    -------------------------------
    {classification_report(y_test, predictions, target_names=['Benign', 'DDoS'])}
    
    Confusion Matrix:
    -----------------
    {confusion_matrix(y_test, predictions)}
    
    """
    
    return report, metrics

def load_and_merge_datasets(file_patterns):
    """
    Load and merge multiple datasets
    
    Args:
        file_patterns: List of file patterns to match
    
    Returns:
        X_merged: Merged feature matrix
        y_merged: Merged label vector
    """
    X_list = []
    y_list = []
    
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for file in files:
            print(f"Loading {file}...")
            with h5py.File(file, 'r') as f:
                X = np.array(f['set_x'][:])
                y = np.array(f['set_y'][:])
                
                # Flatten if needed
                if len(X.shape) > 2:
                    X = X.reshape(X.shape[0], -1)
                
                X_list.append(X)
                y_list.append(y)
    
    # Merge all datasets
    X_merged = np.vstack(X_list)
    y_merged = np.hstack(y_list)
    
    print(f"Merged dataset: {X_merged.shape[0]:,} samples, {X_merged.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_merged.astype(int))}")
    
    return X_merged, y_merged

def generate_synthetic_adversarial_samples(X, y, model, epsilon=0.01, num_samples=1000):
    """
    Generate simple adversarial samples for robustness testing
    """
    # Select malicious samples
    malicious_indices = np.where(y == 1)[0]
    if len(malicious_indices) < num_samples:
        selected_indices = malicious_indices
    else:
        selected_indices = np.random.choice(malicious_indices, num_samples, replace=False)
    
    X_adv = X[selected_indices].copy()
    y_adv = y[selected_indices].copy()
    
    # Add small random noise
    noise = np.random.normal(0, epsilon, X_adv.shape)
    X_adv += noise
    
    # Ensure values stay within reasonable bounds
    X_adv = np.clip(X_adv, 0, np.max(X))
    
    return X_adv, y_adv
