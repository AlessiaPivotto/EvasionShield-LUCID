#!/usr/bin/env python3
"""
Feature Reset (Zeroing) Evaluation for LUCID DDoS Detection
===========================================================

This script evaluates the impact of feature removal on LUCID model performance
by systematically zeroing out individual features or groups of features.

Author: EvasionShield-LUCID Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import os
import h5py
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add parent directories to path
sys.path.append('../')
sys.path.append('../lucid-ddos-master')
sys.path.append('../flatten_lucid')
sys.path.append('../multiclass-lucid')

try:
    from lucid_dataset_parser import load_dataset
    from util_functions import *
except ImportError as e:
    print(f"Warning: Could not import LUCID utilities: {e}")
    print("Make sure you're running from the correct directory")

class FeatureResetEvaluator:
    """
    Evaluates model performance by systematically resetting (zeroing) features.
    """
    
    def __init__(self, model_path, test_data_path, output_dir="./feature_reset_results"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained LUCID model (.h5 file)
            test_data_path: Path to test dataset (.hdf5 file)
            output_dir: Directory to save results
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load model and test data
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        
        print(f"Loading test data from: {test_data_path}")
        self.X_test, self.Y_test = self._load_test_data(test_data_path)
        
        # Get baseline performance
        print("Computing baseline performance...")
        self.baseline_performance = self._evaluate_model(self.X_test, self.Y_test)
        print(f"Baseline Accuracy: {self.baseline_performance['accuracy']:.4f}")
        print(f"Baseline F1-Score: {self.baseline_performance['f1_score']:.4f}")
        
        # Determine model type and feature structure
        self.model_type = self._determine_model_type()
        self.feature_names = self._get_feature_names()
        
        print(f"Model type: {self.model_type}")
        print(f"Input shape: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
    
    def _load_test_data(self, data_path):
        """Load test data from HDF5 file."""
        try:
            return load_dataset(data_path)
        except Exception as e:
            print(f"Error loading with LUCID parser: {e}")
            print("Trying direct HDF5 loading...")
            
            with h5py.File(data_path, 'r') as f:
                X = f['x'][:]
                Y = f['y'][:]
                return X, Y
    
    def _determine_model_type(self):
        """Determine if model is CNN, flatten, or multiclass."""
        input_shape = self.X_test.shape
        
        if len(input_shape) == 4:  # (samples, height, width, channels)
            return "CNN"
        elif len(input_shape) == 2:  # (samples, features)
            return "Flatten"
        else:
            return "Unknown"
    
    def _get_feature_names(self):
        """Get feature names based on model type."""
        if self.model_type == "CNN":
            # For CNN, features are packet-level in each time window
            height, width = self.X_test.shape[1:3]
            return [f"Packet_{i}_Feature_{j}" for i in range(height) for j in range(width)]
        elif self.model_type == "Flatten":
            # For flatten model, features are statistical aggregations
            num_features = self.X_test.shape[1]
            return [f"Statistical_Feature_{i}" for i in range(num_features)]
        else:
            return None
    
    def _evaluate_model(self, X, Y, threshold=0.5):
        """Evaluate model performance on given data."""
        predictions = self.model.predict(X, batch_size=2048, verbose=0)
        
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multiclass
            y_pred = np.argmax(predictions, axis=1)
            y_true = Y if len(Y.shape) == 1 else np.argmax(Y, axis=1)
        else:
            # Binary classification
            y_pred = (predictions > threshold).astype(int).flatten()
            y_true = Y.flatten()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def evaluate_individual_features(self, save_results=True):
        """
        Evaluate impact of zeroing individual features.
        
        Returns:
            DataFrame with results for each feature
        """
        print("Evaluating individual feature importance...")
        results = []
        
        if self.model_type == "CNN":
            height, width = self.X_test.shape[1:3]
            total_features = height * width
            
            for i in range(height):
                for j in range(width):
                    print(f"Processing feature ({i}, {j}) - {(i*width + j + 1)}/{total_features}")
                    
                    # Create modified data with feature zeroed
                    X_modified = self.X_test.copy()
                    X_modified[:, i, j] = 0
                    
                    # Evaluate performance
                    performance = self._evaluate_model(X_modified, self.Y_test)
                    
                    # Calculate impact
                    accuracy_drop = self.baseline_performance['accuracy'] - performance['accuracy']
                    f1_drop = self.baseline_performance['f1_score'] - performance['f1_score']
                    
                    results.append({
                        'feature_index': f"({i},{j})",
                        'feature_name': f"Packet_{i}_Feature_{j}",
                        'accuracy': performance['accuracy'],
                        'f1_score': performance['f1_score'],
                        'accuracy_drop': accuracy_drop,
                        'f1_drop': f1_drop,
                        'importance_score': (accuracy_drop + f1_drop) / 2
                    })
        
        elif self.model_type == "Flatten":
            num_features = self.X_test.shape[1]
            
            for i in range(num_features):
                print(f"Processing feature {i+1}/{num_features}")
                
                # Create modified data with feature zeroed
                X_modified = self.X_test.copy()
                X_modified[:, i] = 0
                
                # Evaluate performance
                performance = self._evaluate_model(X_modified, self.Y_test)
                
                # Calculate impact
                accuracy_drop = self.baseline_performance['accuracy'] - performance['accuracy']
                f1_drop = self.baseline_performance['f1_score'] - performance['f1_score']
                
                results.append({
                    'feature_index': i,
                    'feature_name': f"Statistical_Feature_{i}",
                    'accuracy': performance['accuracy'],
                    'f1_score': performance['f1_score'],
                    'accuracy_drop': accuracy_drop,
                    'f1_drop': f1_drop,
                    'importance_score': (accuracy_drop + f1_drop) / 2
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('importance_score', ascending=False)
        
        if save_results:
            output_file = self.output_dir / "individual_feature_importance.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
        return results_df
    
    def evaluate_feature_groups(self, group_configs=None, save_results=True):
        """
        Evaluate impact of zeroing groups of features.
        
        Args:
            group_configs: List of dictionaries defining feature groups
            save_results: Whether to save results to file
        """
        if group_configs is None:
            group_configs = self._get_default_group_configs()
        
        print("Evaluating feature group importance...")
        results = []
        
        for group_config in group_configs:
            group_name = group_config['name']
            print(f"Processing group: {group_name}")
            
            # Create modified data with group zeroed
            X_modified = self._zero_feature_group(self.X_test.copy(), group_config)
            
            # Evaluate performance
            performance = self._evaluate_model(X_modified, self.Y_test)
            
            # Calculate impact
            accuracy_drop = self.baseline_performance['accuracy'] - performance['accuracy']
            f1_drop = self.baseline_performance['f1_score'] - performance['f1_score']
            
            results.append({
                'group_name': group_name,
                'group_description': group_config.get('description', ''),
                'accuracy': performance['accuracy'],
                'f1_score': performance['f1_score'],
                'accuracy_drop': accuracy_drop,
                'f1_drop': f1_drop,
                'importance_score': (accuracy_drop + f1_drop) / 2
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('importance_score', ascending=False)
        
        if save_results:
            output_file = self.output_dir / "feature_group_importance.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
        return results_df
    
    def _get_default_group_configs(self):
        """Get default feature group configurations."""
        if self.model_type == "CNN":
            height, width = self.X_test.shape[1:3]
            return [
                {
                    'name': 'first_half_packets',
                    'description': 'First half of packets in time window',
                    'type': 'rows',
                    'indices': list(range(height // 2))
                },
                {
                    'name': 'second_half_packets',
                    'description': 'Second half of packets in time window',
                    'type': 'rows',
                    'indices': list(range(height // 2, height))
                },
                {
                    'name': 'early_features',
                    'description': 'Early features (first columns)',
                    'type': 'columns',
                    'indices': list(range(width // 2))
                },
                {
                    'name': 'late_features',
                    'description': 'Late features (last columns)',
                    'type': 'columns',
                    'indices': list(range(width // 2, width))
                }
            ]
        
        elif self.model_type == "Flatten":
            num_features = self.X_test.shape[1]
            return [
                {
                    'name': 'first_quarter',
                    'description': 'First quarter of features',
                    'type': 'features',
                    'indices': list(range(num_features // 4))
                },
                {
                    'name': 'second_quarter',
                    'description': 'Second quarter of features',
                    'type': 'features',
                    'indices': list(range(num_features // 4, num_features // 2))
                },
                {
                    'name': 'third_quarter',
                    'description': 'Third quarter of features',
                    'type': 'features',
                    'indices': list(range(num_features // 2, 3 * num_features // 4))
                },
                {
                    'name': 'fourth_quarter',
                    'description': 'Fourth quarter of features',
                    'type': 'features',
                    'indices': list(range(3 * num_features // 4, num_features))
                }
            ]
        
        return []
    
    def _zero_feature_group(self, X, group_config):
        """Zero out a group of features according to configuration."""
        if group_config['type'] == 'rows':
            for idx in group_config['indices']:
                if idx < X.shape[1]:
                    X[:, idx, :] = 0
        elif group_config['type'] == 'columns':
            for idx in group_config['indices']:
                if idx < X.shape[2]:
                    X[:, :, idx] = 0
        elif group_config['type'] == 'features':
            for idx in group_config['indices']:
                if idx < X.shape[1]:
                    X[:, idx] = 0
        
        return X
    
    def progressive_feature_removal(self, max_features=20, save_results=True):
        """
        Progressively remove most important features and evaluate performance.
        
        Args:
            max_features: Maximum number of features to remove
            save_results: Whether to save results to file
        """
        print("Running progressive feature removal analysis...")
        
        # First get individual feature importance
        individual_results = self.evaluate_individual_features(save_results=False)
        top_features = individual_results.head(max_features)
        
        results = []
        X_modified = self.X_test.copy()
        removed_features = []
        
        # Add baseline
        results.append({
            'features_removed': 0,
            'removed_feature_names': '',
            'accuracy': self.baseline_performance['accuracy'],
            'f1_score': self.baseline_performance['f1_score']
        })
        
        for idx, row in top_features.iterrows():
            # Zero out the feature
            if self.model_type == "CNN":
                # Parse indices from string like "(1,2)"
                indices = row['feature_index'].strip('()').split(',')
                i, j = int(indices[0]), int(indices[1])
                X_modified[:, i, j] = 0
            elif self.model_type == "Flatten":
                feature_idx = row['feature_index']
                X_modified[:, feature_idx] = 0
            
            removed_features.append(row['feature_name'])
            
            # Evaluate performance
            performance = self._evaluate_model(X_modified, self.Y_test)
            
            results.append({
                'features_removed': len(removed_features),
                'removed_feature_names': ', '.join(removed_features),
                'accuracy': performance['accuracy'],
                'f1_score': performance['f1_score']
            })
            
            print(f"Removed {len(removed_features)} features - "
                  f"Accuracy: {performance['accuracy']:.4f}, "
                  f"F1: {performance['f1_score']:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if save_results:
            output_file = self.output_dir / "progressive_removal.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
        return results_df
    
    def create_visualizations(self, individual_df=None, group_df=None, progressive_df=None):
        """Create comprehensive visualizations of feature importance results."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Individual feature importance visualization
        if individual_df is not None:
            self._plot_individual_importance(individual_df)
        
        # Feature group importance visualization
        if group_df is not None:
            self._plot_group_importance(group_df)
        
        # Progressive removal visualization
        if progressive_df is not None:
            self._plot_progressive_removal(progressive_df)
        
        # Combined summary plot
        self._plot_summary(individual_df, group_df, progressive_df)
    
    def _plot_individual_importance(self, df):
        """Plot individual feature importance."""
        top_features = df.head(20)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy drop
        ax1.barh(range(len(top_features)), top_features['accuracy_drop'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature_name'], fontsize=8)
        ax1.set_xlabel('Accuracy Drop')
        ax1.set_title('Top 20 Features by Accuracy Impact')
        ax1.invert_yaxis()
        
        # F1 drop
        ax2.barh(range(len(top_features)), top_features['f1_drop'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature_name'], fontsize=8)
        ax2.set_xlabel('F1-Score Drop')
        ax2.set_title('Top 20 Features by F1-Score Impact')
        ax2.invert_yaxis()
        
        # Importance score distribution
        ax3.hist(df['importance_score'], bins=50, alpha=0.7)
        ax3.set_xlabel('Importance Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Feature Importance Scores')
        ax3.axvline(df['importance_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["importance_score"].mean():.4f}')
        ax3.legend()
        
        # Correlation between accuracy and F1 drops
        ax4.scatter(df['accuracy_drop'], df['f1_drop'], alpha=0.6)
        ax4.set_xlabel('Accuracy Drop')
        ax4.set_ylabel('F1-Score Drop')
        ax4.set_title('Correlation: Accuracy vs F1-Score Impact')
        
        # Add correlation coefficient
        correlation = np.corrcoef(df['accuracy_drop'], df['f1_drop'])[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "individual_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_group_importance(self, df):
        """Plot feature group importance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group importance scores
        ax1.barh(range(len(df)), df['importance_score'])
        ax1.set_yticks(range(len(df)))
        ax1.set_yticklabels(df['group_name'])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Feature Group Importance')
        ax1.invert_yaxis()
        
        # Accuracy vs F1 drop for groups
        ax2.scatter(df['accuracy_drop'], df['f1_drop'], s=100)
        for i, txt in enumerate(df['group_name']):
            ax2.annotate(txt, (df['accuracy_drop'].iloc[i], df['f1_drop'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Accuracy Drop')
        ax2.set_ylabel('F1-Score Drop')
        ax2.set_title('Group Impact: Accuracy vs F1-Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_group_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_progressive_removal(self, df):
        """Plot progressive feature removal results."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Performance degradation (scale 0-1)
        baseline_acc = df['accuracy'].iloc[0]
        baseline_f1 = df['f1_score'].iloc[0]
        acc_degradation = (baseline_acc - df['accuracy']) / baseline_acc
        f1_degradation = (baseline_f1 - df['f1_score']) / baseline_f1
        
        ax.plot(df['features_removed'], acc_degradation, 'o-', label='Accuracy Degradation', linewidth=2)
        ax.plot(df['features_removed'], f1_degradation, 's-', label='F1-Score Degradation', linewidth=2)
        ax.set_xlabel('Number of Features Removed')
        ax.set_ylabel('Performance Degradation (0-1 scale)')
        ax.set_title('Performance Degradation vs Feature Removal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "progressive_removal.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary(self, individual_df, group_df, progressive_df):
        """Create a summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top 10 individual features
        if individual_df is not None:
            top10 = individual_df.head(10)
            ax1.barh(range(len(top10)), top10['importance_score'])
            ax1.set_yticks(range(len(top10)))
            ax1.set_yticklabels(top10['feature_name'], fontsize=8)
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top 10 Individual Features')
            ax1.invert_yaxis()
        
        # Group comparison
        if group_df is not None:
            ax2.bar(group_df['group_name'], group_df['importance_score'])
            ax2.set_ylabel('Importance Score')
            ax2.set_title('Feature Group Comparison')
            ax2.tick_params(axis='x', rotation=45)
        
        # Performance degradation overview
        if progressive_df is not None:
            baseline_acc = progressive_df['accuracy'].iloc[0]
            baseline_f1 = progressive_df['f1_score'].iloc[0]
            
            # Find point where performance drops by 5%
            acc_5percent = baseline_acc * 0.95
            f1_5percent = baseline_f1 * 0.95
            
            ax3.plot(progressive_df['features_removed'], progressive_df['accuracy'], 
                    'o-', label='Accuracy', linewidth=2)
            ax3.plot(progressive_df['features_removed'], progressive_df['f1_score'], 
                    's-', label='F1-Score', linewidth=2)
            ax3.axhline(y=acc_5percent, color='red', linestyle='--', alpha=0.7, label='5% drop')
            ax3.axhline(y=f1_5percent, color='red', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Features Removed')
            ax3.set_ylabel('Performance')
            ax3.set_title('Progressive Removal Overview')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Feature importance statistics
        if individual_df is not None:
            stats_text = f"""
            Feature Importance Statistics:
            
            Total Features: {len(individual_df)}
            Mean Importance: {individual_df['importance_score'].mean():.4f}
            Std Importance: {individual_df['importance_score'].std():.4f}
            Max Importance: {individual_df['importance_score'].max():.4f}
            
            High Impact Features (>2 std): {len(individual_df[individual_df['importance_score'] > individual_df['importance_score'].mean() + 2*individual_df['importance_score'].std()])}
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, individual_df, group_df, progressive_df):
        """Generate a comprehensive report."""
        report_path = self.output_dir / "feature_reset_evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Feature Reset Evaluation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model**: {self.model_path}\n\n")
            f.write(f"**Test Data**: {self.test_data_path}\n\n")
            
            # Baseline performance
            f.write("## Baseline Performance\n\n")
            f.write(f"- **Accuracy**: {self.baseline_performance['accuracy']:.4f}\n")
            f.write(f"- **F1-Score**: {self.baseline_performance['f1_score']:.4f}\n")
            f.write(f"- **Precision**: {self.baseline_performance['precision']:.4f}\n")
            f.write(f"- **Recall**: {self.baseline_performance['recall']:.4f}\n\n")
            
            # Model information
            f.write("## Model Information\n\n")
            f.write(f"- **Model Type**: {self.model_type}\n")
            f.write(f"- **Input Shape**: {self.X_test.shape}\n")
            f.write(f"- **Number of Test Samples**: {len(self.Y_test)}\n\n")
            
            # Individual feature analysis
            if individual_df is not None:
                f.write("## Individual Feature Analysis\n\n")
                f.write(f"- **Total Features Analyzed**: {len(individual_df)}\n")
                f.write(f"- **Mean Importance Score**: {individual_df['importance_score'].mean():.4f}\n")
                f.write(f"- **Std Importance Score**: {individual_df['importance_score'].std():.4f}\n")
                f.write(f"- **Most Important Feature**: {individual_df.iloc[0]['feature_name']} (Score: {individual_df.iloc[0]['importance_score']:.4f})\n\n")
                
                f.write("### Top 10 Most Important Features\n\n")
                top10 = individual_df.head(10)
                f.write("| Rank | Feature | Accuracy Drop | F1 Drop | Importance Score |\n")
                f.write("|------|---------|---------------|---------|------------------|\n")
                for idx, row in top10.iterrows():
                    f.write(f"| {idx+1} | {row['feature_name']} | {row['accuracy_drop']:.4f} | {row['f1_drop']:.4f} | {row['importance_score']:.4f} |\n")
                f.write("\n")
            
            # Group analysis
            if group_df is not None:
                f.write("## Feature Group Analysis\n\n")
                f.write("| Group | Description | Accuracy Drop | F1 Drop | Importance Score |\n")
                f.write("|-------|-------------|---------------|---------|------------------|\n")
                for _, row in group_df.iterrows():
                    f.write(f"| {row['group_name']} | {row['group_description']} | {row['accuracy_drop']:.4f} | {row['f1_drop']:.4f} | {row['importance_score']:.4f} |\n")
                f.write("\n")
            
            # Progressive removal analysis
            if progressive_df is not None:
                f.write("## Progressive Removal Analysis\n\n")
                baseline_acc = progressive_df['accuracy'].iloc[0]
                baseline_f1 = progressive_df['f1_score'].iloc[0]
                
                # Find critical points
                acc_5_drop = baseline_acc * 0.95
                f1_5_drop = baseline_f1 * 0.95
                
                acc_critical = progressive_df[progressive_df['accuracy'] <= acc_5_drop]
                f1_critical = progressive_df[progressive_df['f1_score'] <= f1_5_drop]
                
                if not acc_critical.empty:
                    f.write(f"- **Accuracy drops by 5% after removing**: {acc_critical.iloc[0]['features_removed']} features\n")
                if not f1_critical.empty:
                    f.write(f"- **F1-Score drops by 5% after removing**: {f1_critical.iloc[0]['features_removed']} features\n")
                
                f.write(f"- **Final performance after removing top {len(progressive_df)-1} features**:\n")
                f.write(f"  - Accuracy: {progressive_df.iloc[-1]['accuracy']:.4f} ({((progressive_df.iloc[-1]['accuracy'] - baseline_acc) / baseline_acc * 100):+.1f}%)\n")
                f.write(f"  - F1-Score: {progressive_df.iloc[-1]['f1_score']:.4f} ({((progressive_df.iloc[-1]['f1_score'] - baseline_f1) / baseline_f1 * 100):+.1f}%)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the feature reset evaluation:\n\n")
            
            if individual_df is not None:
                high_impact = individual_df[individual_df['importance_score'] > individual_df['importance_score'].mean() + 2*individual_df['importance_score'].std()]
                f.write(f"1. **Critical Features**: {len(high_impact)} features have high impact (>2 standard deviations above mean)\n")
                
                low_impact = individual_df[individual_df['importance_score'] < individual_df['importance_score'].mean() - individual_df['importance_score'].std()]
                f.write(f"2. **Potential for Feature Reduction**: {len(low_impact)} features have low impact and could potentially be removed\n")
            
            f.write("3. **Model Robustness**: Evaluate the model's sensitivity to feature perturbations\n")
            f.write("4. **Feature Engineering**: Consider focusing on the most important features for future improvements\n")
            f.write("5. **Adversarial Robustness**: High-impact features may be targets for adversarial attacks\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `individual_feature_importance.csv`: Detailed individual feature analysis\n")
            f.write("- `feature_group_importance.csv`: Feature group analysis\n")
            f.write("- `progressive_removal.csv`: Progressive feature removal results\n")
            f.write("- `individual_feature_importance.png`: Individual feature visualizations\n")
            f.write("- `feature_group_importance.png`: Group importance visualizations\n")
            f.write("- `progressive_removal.png`: Progressive removal plots\n")
            f.write("- `feature_importance_summary.png`: Summary visualization\n")
        
        print(f"Report generated: {report_path}")
    
    def run_complete_evaluation(self, max_progressive_features=20):
        """Run the complete feature reset evaluation."""
        print("Starting complete feature reset evaluation...")
        
        # Individual feature evaluation
        individual_results = self.evaluate_individual_features()
        
        # Feature group evaluation
        group_results = self.evaluate_feature_groups()
        
        # Progressive removal evaluation
        progressive_results = self.progressive_feature_removal(max_features=max_progressive_features)
        
        # Create visualizations
        self.create_visualizations(individual_results, group_results, progressive_results)
        
        # Generate report
        self.generate_report(individual_results, group_results, progressive_results)
        
        print(f"\nComplete evaluation finished! Results saved in: {self.output_dir}")
        return individual_results, group_results, progressive_results


def main():
    parser = argparse.ArgumentParser(description="Feature Reset Evaluation for LUCID")
    parser.add_argument('-m', '--model', required=True, help="Path to trained model (.h5)")
    parser.add_argument('-d', '--data', required=True, help="Path to test dataset (.hdf5)")
    parser.add_argument('-o', '--output', default="./feature_reset_results", 
                       help="Output directory for results")
    parser.add_argument('--max_progressive', type=int, default=20,
                       help="Maximum features for progressive removal")
    parser.add_argument('--individual_only', action='store_true',
                       help="Run only individual feature evaluation")
    parser.add_argument('--groups_only', action='store_true',
                       help="Run only feature group evaluation")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FeatureResetEvaluator(args.model, args.data, args.output)
    
    if args.individual_only:
        individual_results = evaluator.evaluate_individual_features()
        evaluator.create_visualizations(individual_df=individual_results)
    elif args.groups_only:
        group_results = evaluator.evaluate_feature_groups()
        evaluator.create_visualizations(group_df=group_results)
    else:
        # Run complete evaluation
        evaluator.run_complete_evaluation(max_progressive_features=args.max_progressive)


if __name__ == "__main__":
    main()
