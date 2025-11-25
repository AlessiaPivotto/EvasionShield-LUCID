#!/usr/bin/env python3
"""
Comprehensive Random Forest Analysis Suite for EvasionShield-LUCID
==================================================================

This unified module provides complete Random Forest functionality for DDoS detection:
- Basic Random Forest binary classification
- Cross-dataset comparison and analysis
- Comprehensive performance evaluation
- Advanced plotting and visualization
- Statistical analysis and reporting

Author: AI Assistant
Project: EvasionShield-LUCID
License: Apache License 2.0

Usage Examples:
--------------
# Basic training and testing
python3 rf_unified.py --train --dataset FLATTEN-PCAPS

# Cross-dataset comparison (main feature)
python3 rf_unified.py --compare-datasets

# Comprehensive analysis with hyperparameter tuning
python3 rf_unified.py --comprehensive --tune

# Single dataset testing
python3 rf_unified.py --test --dataset MANIPULATED-FLATTEN_42 --model saved_model.pkl
"""

import os
import sys
import glob
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from datetime import datetime
from pathlib import Path
from scipy import stats

# Scientific computing and ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, 
                            roc_curve, auc, precision_recall_curve)
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# Add paths for utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'flatten_lucid'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lucid-ddos-master'))

try:
    from util_functions import SEED
    SEED_AVAILABLE = True
    print("✅ Successfully imported util_functions")
except ImportError:
    SEED = 42
    SEED_AVAILABLE = False
    print("⚠️  util_functions not found, using default SEED=42")

# Set random seeds for reproducibility
np.random.seed(SEED)


class RandomForestUnifiedAnalysis:
    """
    Unified Random Forest Analysis Suite
    
    Combines all Random Forest functionality into a single comprehensive class:
    - Basic binary classification
    - Cross-dataset comparison
    - Performance analysis
    - Advanced visualization
    - Statistical reporting
    """
    
    def __init__(self, base_datasets_dir="./TrafficManipulator-master/DATASETS", 
                 output_dir="./RandomForest/results", 
                 model_save_path="./RandomForest/models"):
        """
        Initialize the unified Random Forest analysis suite
        
        Args:
            base_datasets_dir: Path to datasets directory
            output_dir: Path for saving results
            model_save_path: Path for saving models
        """
        self.base_datasets_dir = base_datasets_dir
        self.output_dir = output_dir
        self.model_save_path = model_save_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Target datasets for comparison
        self.target_datasets = [
            'FLATTEN-PCAPS',
            'MANIPULATED-FLATTEN_42', 
            'FRAGMENTED_42_150'
        ]
        
        # All available datasets
        self.all_datasets = [
            'FLATTEN-PCAPS',
            'MANIPULATED-FLATTEN_0',
            'MANIPULATED-FLATTEN_12345',
            'MANIPULATED-FLATTEN_42',
            'FRAGMENTED_42_150',
            'FRAGMENTED_42_500',
            'PCAP-CHUNKS'
        ]
        
        # Storage for models and results
        self.models = {}
        self.results = []
        self.training_history = {}
        
        # Default RF parameters
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': SEED,
            'n_jobs': -1
        }

    # ==================== DATA LOADING ====================
    
    def load_hdf5_data(self, file_path):
        """Load data from HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                X = np.array(f['set_x'][:])
                Y = np.array(f['set_y'][:])
            print(f"✅ Loaded {X.shape[0]} samples, {X.shape[1]} features from {os.path.basename(file_path)}")
            return X, Y
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            return None, None
    
    def get_attack_type_from_folder(self, folder_name):
        """Extract attack type from folder name"""
        attack_mapping = {
            'PortMap': 'PortMap', 'Portmap': 'PortMap',
            'NetBIOS': 'NetBIOS', 
            'LDAP': 'LDAP',
            'MSSQL': 'MSSQL',
            'UDP': 'UDP',
            'SYN': 'SYN', 'Syn': 'SYN',
            'UDPLag': 'UDPLag',
            'WebDDos': 'WebDDoS', 'WebDDoS': 'WebDDoS',
            'TFTP': 'TFTP',
            'NTP': 'NTP',
            'SSDP': 'SSDP',
            'DNS': 'DNS',
            'SNMP': 'SNMP'
        }
        
        folder_lower = folder_name.lower()
        for key, value in attack_mapping.items():
            if key.lower() in folder_lower:
                return value
        
        return folder_name

    # ==================== MODEL TRAINING ====================
    
    def create_model(self, params=None):
        """Create Random Forest model with specified parameters"""
        model_params = self.default_params.copy()
        if params:
            model_params.update(params)
        
        return RandomForestClassifier(**model_params)
    
    def train_model_on_dataset(self, dataset_name, tune_hyperparams=False):
        """Train Random Forest model on a specific dataset"""
        print(f"\n🔧 Training Random Forest on {dataset_name}...")
        
        dataset_path = os.path.join(self.base_datasets_dir, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None
            
        # Find training file
        train_files = glob.glob(os.path.join(dataset_path, "*train*.hdf5"))
        if not train_files:
            print(f"❌ No training file found in {dataset_path}")
            return None
            
        train_file = train_files[0]
        print(f"📂 Using training file: {os.path.basename(train_file)}")
        
        # Load training data
        X_train, y_train = self.load_hdf5_data(train_file)
        if X_train is None:
            return None
            
        # Preprocess labels
        y_train = y_train.flatten()
        if len(np.unique(y_train)) == 2:
            y_train = (y_train > 0).astype(int)
        
        print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"📊 Class distribution: {np.bincount(y_train)}")
        
        # Create model
        if tune_hyperparams:
            model = self.hyperparameter_tuning(X_train, y_train)
        else:
            model = self.create_model()
        
        # Train model
        print("🚀 Training Random Forest...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Store training info
        self.training_history[dataset_name] = {
            'training_time': training_time,
            'samples': X_train.shape[0],
            'features': X_train.shape[1],
            'class_distribution': np.bincount(y_train).tolist()
        }
        
        # Store model
        self.models[dataset_name] = model
        
        return model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        print("🎯 Performing hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, 
                                 scoring='f1', n_jobs=-1, verbose=1)
        
        grid_search.fit(X_train, y_train)
        
        print(f"🏆 Best parameters: {grid_search.best_params_}")
        print(f"🎯 Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_

    # ==================== MODEL TESTING ====================
    
    def test_model_on_dataset(self, model, model_name, test_dataset):
        """Test a model on a specific dataset"""
        print(f"\n🧪 Testing {model_name} model on {test_dataset} data...")
        
        test_path = os.path.join(self.base_datasets_dir, test_dataset)
        
        # Get all test folders
        test_folders = []
        for item in os.listdir(test_path):
            item_path = os.path.join(test_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                test_folders.append(item)
        
        dataset_results = []
        
        for folder in sorted(test_folders):
            folder_path = os.path.join(test_path, folder)
            test_files = glob.glob(os.path.join(folder_path, "*.hdf5"))
            
            if not test_files:
                continue
                
            print(f"  📁 Testing on folder: {folder}")
            
            folder_metrics = []
            
            for test_file in test_files:
                X_test, y_test = self.load_hdf5_data(test_file)
                if X_test is None:
                    continue
                
                # Preprocess labels
                y_test = y_test.flatten()
                if len(np.unique(y_test)) == 2:
                    y_test = (y_test > 0).astype(int)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
                folder_metrics.append(metrics)
            
            if folder_metrics:
                # Average metrics for this folder
                avg_metrics = {}
                for metric in folder_metrics[0].keys():
                    avg_metrics[metric] = np.mean([m[metric] for m in folder_metrics])
                
                # Store result
                result = {
                    'model_name': model_name,
                    'test_dataset': test_dataset,
                    'folder': folder,
                    'attack_type': self.get_attack_type_from_folder(folder),
                    **avg_metrics
                }
                
                dataset_results.append(result)
                self.results.append(result)
                
                print(f"    📊 F1: {avg_metrics['f1_score']:.3f}, FNR: {avg_metrics['fnr']:.3f}")
        
        return dataset_results
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive performance metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fnr': fnr,
            'fpr': fpr,
            'tnr': tnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_true, y_pred_proba)
                metrics['auc'] = auc_score
            except ValueError:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
            
        return metrics

    # ==================== COMPARISON ANALYSIS ====================
    
    def run_cross_dataset_comparison(self):
        """Run cross-dataset comparison analysis"""
        print("🚀 Starting Cross-Dataset Comparison Analysis")
        print(f"📊 Target datasets: {', '.join(self.target_datasets)}")
        
        # Train model on FLATTEN-PCAPS (reference dataset)
        print("\n🔧 Training reference model on FLATTEN-PCAPS...")
        model = self.train_model_on_dataset('FLATTEN-PCAPS')
        if model is None:
            print("❌ Failed to train reference model")
            return
        
        # Test on all target datasets
        for test_dataset in self.target_datasets:
            self.test_model_on_dataset(model, 'FLATTEN-PCAPS', test_dataset)
        
        # Generate comprehensive results
        self.generate_comparison_results()
    
    def run_comprehensive_analysis(self, tune_hyperparams=False):
        """Run comprehensive analysis across all available datasets"""
        print("🚀 Starting Comprehensive Multi-Dataset Analysis")
        
        # Train models on datasets that have training data
        training_datasets = []
        for dataset in self.all_datasets:
            dataset_path = os.path.join(self.base_datasets_dir, dataset)
            if os.path.exists(dataset_path):
                train_files = glob.glob(os.path.join(dataset_path, "*train*.hdf5"))
                if train_files:
                    training_datasets.append(dataset)
                    self.train_model_on_dataset(dataset, tune_hyperparams)
        
        print(f"\n✅ Successfully trained models on: {training_datasets}")
        
        # Cross-test all models on all datasets
        for model_dataset in training_datasets:
            if model_dataset not in self.models:
                continue
                
            model = self.models[model_dataset]
            
            for test_dataset in self.all_datasets:
                test_path = os.path.join(self.base_datasets_dir, test_dataset)
                if os.path.exists(test_path):
                    self.test_model_on_dataset(model, model_dataset, test_dataset)
        
        # Generate comprehensive results
        self.generate_comprehensive_results()

    # ==================== VISUALIZATION ====================
    
    def create_comparison_plots(self, df, plot_type="comparison"):
        """Create comprehensive comparison plots"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        if plot_type == "comparison":
            return self._create_cross_dataset_plots(df)
        elif plot_type == "comprehensive":
            return self._create_comprehensive_plots(df)
    
    def _create_cross_dataset_plots(self, df):
        """Create cross-dataset comparison plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Random Forest Cross-Dataset Comparison Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        attack_types = sorted(df['attack_type'].unique())
        datasets = sorted(df['test_dataset'].unique())
        
        # 1. F1 Score Heatmap
        pivot_f1 = df.pivot_table(index='attack_type', columns='test_dataset', 
                                  values='f1_score', aggfunc='mean')
        pivot_f1 = pivot_f1.reindex(attack_types)
        
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='Blues', ax=ax1, 
                    cbar_kws={'label': 'F1 Score'}, linewidths=0.5, linecolor='white')
        ax1.set_title('F1 Score Heatmap by Dataset', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        # 2. FNR Heatmap (main plot)
        pivot_fnr = df.pivot_table(index='attack_type', columns='test_dataset', 
                                   values='fnr', aggfunc='mean')
        pivot_fnr = pivot_fnr.reindex(attack_types)
        
        sns.heatmap(pivot_fnr, annot=True, fmt='.3f', cmap='Reds', ax=ax2, 
                    cbar_kws={'label': 'False Negative Rate'}, linewidths=0.5, linecolor='white')
        ax2.set_title('False Negative Rate Heatmap by Dataset', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        # 3. Performance degradation analysis
        if 'FLATTEN-PCAPS' in datasets:
            baseline_f1 = df[df['test_dataset'] == 'FLATTEN-PCAPS'].groupby('attack_type')['f1_score'].mean()
            
            degradation_data = []
            for dataset in datasets:
                if dataset != 'FLATTEN-PCAPS':
                    dataset_f1 = df[df['test_dataset'] == dataset].groupby('attack_type')['f1_score'].mean()
                    for attack_type in attack_types:
                        if attack_type in baseline_f1.index and attack_type in dataset_f1.index:
                            degradation = baseline_f1[attack_type] - dataset_f1[attack_type]
                            degradation_data.append({
                                'attack_type': attack_type,
                                'dataset': dataset,
                                'degradation': degradation
                            })
            
            if degradation_data:
                deg_df = pd.DataFrame(degradation_data)
                pivot_deg = deg_df.pivot(index='attack_type', columns='dataset', values='degradation')
                
                sns.heatmap(pivot_deg, annot=True, fmt='.3f', cmap='OrRd', ax=ax3,
                            cbar_kws={'label': 'F1 Score Degradation'}, linewidths=0.5, linecolor='white')
                ax3.set_title('Performance Degradation vs FLATTEN-PCAPS', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        # 4. Distribution analysis
        sns.boxplot(data=df, x='test_dataset', y='f1_score', ax=ax4)
        ax4.set_title('F1 Score Distribution by Test Dataset', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
        ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add statistical annotations
        for i, dataset in enumerate(datasets):
            dataset_scores = df[df['test_dataset'] == dataset]['f1_score']
            mean_score = dataset_scores.mean()
            std_score = dataset_scores.std()
            ax4.text(i, mean_score + std_score + 0.02, f'μ={mean_score:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _create_comprehensive_plots(self, df):
        """Create comprehensive analysis plots"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        fig.suptitle('Comprehensive Random Forest Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Implementation of comprehensive plots...
        # (This would include more detailed analysis plots)
        
        plt.tight_layout()
        return fig

    # ==================== RESULTS GENERATION ====================
    
    def generate_comparison_results(self):
        """Generate results for cross-dataset comparison"""
        if not self.results:
            print("❌ No results to generate")
            return
        
        print("\n📊 Generating comparison results and plots...")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_file = os.path.join(self.output_dir, f'rf_cross_dataset_comparison_{self.timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"💾 Results saved to: {csv_file}")
        
        # Generate plots
        fig = self.create_comparison_plots(df, "comparison")
        plot_file = os.path.join(self.output_dir, f'rf_cross_dataset_comparison_{self.timestamp}.png')
        fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"📊 Comparison plot saved: {plot_file}")
        
        # Generate summary statistics
        self.generate_summary_statistics(df)
    
    def generate_comprehensive_results(self):
        """Generate results for comprehensive analysis"""
        if not self.results:
            print("❌ No results to generate")
            return
        
        print("\n📊 Generating comprehensive results...")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed CSV
        csv_file = os.path.join(self.output_dir, f'rf_comprehensive_analysis_{self.timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"💾 Comprehensive results saved to: {csv_file}")
        
        # Generate comprehensive plots
        fig = self.create_comparison_plots(df, "comprehensive")
        plot_file = os.path.join(self.output_dir, f'rf_comprehensive_analysis_{self.timestamp}.png')
        fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"📊 Comprehensive plot saved: {plot_file}")
        
        # Generate detailed report
        self.generate_detailed_report(df)
    
    def generate_summary_statistics(self, df):
        """Generate and print summary statistics"""
        print("\n📈 SUMMARY STATISTICS")
        print("=" * 60)
        
        # Overall performance by dataset
        print("\n🎯 AVERAGE PERFORMANCE BY TEST DATASET:")
        for dataset in df['test_dataset'].unique():
            dataset_df = df[df['test_dataset'] == dataset]
            print(f"\n📊 {dataset}:")
            print(f"  F1 Score: {dataset_df['f1_score'].mean():.3f} ± {dataset_df['f1_score'].std():.3f}")
            print(f"  FNR:      {dataset_df['fnr'].mean():.3f} ± {dataset_df['fnr'].std():.3f}")
            print(f"  Accuracy: {dataset_df['accuracy'].mean():.3f} ± {dataset_df['accuracy'].std():.3f}")
        
        # Best and worst performing combinations
        print(f"\n🏆 BEST PERFORMING COMBINATIONS:")
        best_f1 = df.loc[df['f1_score'].idxmax()]
        print(f"  Highest F1: {best_f1['f1_score']:.3f} ({best_f1['model_name']} → {best_f1['test_dataset']}, {best_f1['attack_type']})")
        
        worst_f1 = df.loc[df['f1_score'].idxmin()]
        print(f"  Lowest F1:  {worst_f1['f1_score']:.3f} ({worst_f1['model_name']} → {worst_f1['test_dataset']}, {worst_f1['attack_type']})")
        
        # Statistical significance tests
        if len(df['test_dataset'].unique()) > 1:
            print(f"\n🔄 STATISTICAL ANALYSIS:")
            datasets = df['test_dataset'].unique()
            for i, dataset1 in enumerate(datasets):
                for dataset2 in datasets[i+1:]:
                    data1 = df[df['test_dataset'] == dataset1]['f1_score']
                    data2 = df[df['test_dataset'] == dataset2]['f1_score']
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        significance = "significant" if p_value < 0.05 else "not significant"
                        print(f"  {dataset1} vs {dataset2}: {significance} (p={p_value:.3f})")
    
    def generate_detailed_report(self, df):
        """Generate detailed markdown report"""
        report_file = os.path.join(self.output_dir, f'rf_detailed_report_{self.timestamp}.md')
        
        with open(report_file, 'w') as f:
            f.write(f"# Random Forest Comprehensive Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Analysis Type:** {'Cross-Dataset Comparison' if len(self.models) == 1 else 'Comprehensive Multi-Model Analysis'}\n\n")
            
            # Dataset summary
            f.write(f"## Datasets Analyzed\n\n")
            for dataset in df['test_dataset'].unique():
                f.write(f"- **{dataset}**\n")
            
            f.write(f"\n## Performance Summary\n\n")
            summary_stats = df.groupby('test_dataset').agg({
                'f1_score': ['mean', 'std', 'min', 'max'],
                'fnr': ['mean', 'std'],
                'accuracy': ['mean', 'std']
            }).round(3)
            
            f.write(f"```\n{summary_stats}\n```\n\n")
            
            # Training info
            if self.training_history:
                f.write(f"## Training Information\n\n")
                for dataset, info in self.training_history.items():
                    f.write(f"### {dataset}\n")
                    f.write(f"- Training time: {info['training_time']:.2f} seconds\n")
                    f.write(f"- Samples: {info['samples']:,}\n")
                    f.write(f"- Features: {info['features']}\n")
                    f.write(f"- Class distribution: {info['class_distribution']}\n\n")
        
        print(f"📋 Detailed report saved: {report_file}")

    # ==================== MODEL PERSISTENCE ====================
    
    def save_model(self, model, model_name):
        """Save trained model"""
        model_file = os.path.join(self.model_save_path, f'{model_name}_{self.timestamp}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Model saved: {model_file}")
        return model_file
    
    def load_model(self, model_path):
        """Load saved model"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded: {model_path}")
        return model

    # ==================== UTILITY FUNCTIONS ====================
    
    def plot_feature_importance(self, model, feature_names=None, top_n=20):
        """Plot feature importance"""
        if not hasattr(model, 'feature_importances_'):
            print("❌ Model does not have feature importance")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        
        if feature_names:
            labels = [feature_names[i] for i in indices]
        else:
            labels = [f'Feature {i}' for i in indices]
        
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), labels, rotation=45, ha='right')
        plt.tight_layout()
        
        return plt
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=['Benign', 'DDoS']):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        return plt


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Unified Random Forest Analysis for EvasionShield-LUCID')
    
    # Main operation modes
    parser.add_argument('--compare-datasets', action='store_true', 
                       help='Run cross-dataset comparison (FLATTEN-PCAPS, MANIPULATED-FLATTEN_42, FRAGMENTED_42_150)')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive analysis across all available datasets')
    parser.add_argument('--train', action='store_true', 
                       help='Train model on specified dataset')
    parser.add_argument('--test', action='store_true', 
                       help='Test model on specified dataset')
    
    # Options
    parser.add_argument('--dataset', type=str, 
                       help='Specify dataset name')
    parser.add_argument('--model', type=str, 
                       help='Path to saved model file')
    parser.add_argument('--tune', action='store_true', 
                       help='Enable hyperparameter tuning')
    parser.add_argument('--output-dir', type=str, default='./RandomForest/results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analysis suite (adjust path if running from RandomForest directory)
    if os.path.basename(os.getcwd()) == 'RandomForest':
        os.chdir('..')
    
    rf_analysis = RandomForestUnifiedAnalysis(output_dir=args.output_dir)
    
    print("🚀 Random Forest Unified Analysis Suite")
    print("=" * 50)
    
    try:
        if args.compare_datasets:
            print("📊 Running Cross-Dataset Comparison...")
            rf_analysis.run_cross_dataset_comparison()
            
        elif args.comprehensive:
            print("🔍 Running Comprehensive Analysis...")
            rf_analysis.run_comprehensive_analysis(tune_hyperparams=args.tune)
            
        elif args.train and args.dataset:
            print(f"🔧 Training model on {args.dataset}...")
            model = rf_analysis.train_model_on_dataset(args.dataset, tune_hyperparams=args.tune)
            if model:
                rf_analysis.save_model(model, args.dataset)
                
        elif args.test and args.dataset and args.model:
            print(f"🧪 Testing model on {args.dataset}...")
            model = rf_analysis.load_model(args.model)
            rf_analysis.test_model_on_dataset(model, 'loaded_model', args.dataset)
            rf_analysis.generate_comparison_results()
            
        else:
            # Default: run cross-dataset comparison
            print("📊 Running default Cross-Dataset Comparison...")
            rf_analysis.run_cross_dataset_comparison()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {rf_analysis.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
