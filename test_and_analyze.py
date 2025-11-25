#!/usr/bin/env python3
"""
Unified Testing and Analysis Script for LUCID DDoS Detection

This script combines testing functionality with comprehensive analysis and visualization.
It allows testing any dataset folder and generates detailed F1 score and FNR analysis.

Usage:
    python3 unified_test_and_analyze.py <dataset_folder1> [dataset_folder2] [...]
    python3 unified_test_and_analyze.py --list-datasets
    python3 unified_test_and_analyze.py --help

Features:
- Tests multiple dataset folders
- Generates F1 score and FNR comparison plots
- Creates comprehensive CSV reports
- Saves all results in organized folder structure
- Command-line dataset specification
- Detailed statistical analysis
"""

import os
import subprocess
import sys
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse
import glob
from scipy import stats

class UnifiedTester:
    def __init__(self, base_datasets_dir="./TrafficManipulator-master/DATASETS", 
                 model_path="./output/10t-100n-DOS2019-LUCID-FLATTEN.h5"):
        self.base_datasets_dir = base_datasets_dir
        self.model_path = model_path
        self.results_base_dir = "./ResultsFlattenLucid"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = None
        
        # Ensure results directory exists
        os.makedirs(self.results_base_dir, exist_ok=True)
        
    def create_run_directory(self, dataset_names):
        """Create a specific directory for this run"""
        run_name = f"run_{self.timestamp}_{'_'.join([name.replace('/', '_') for name in dataset_names])}"
        self.run_dir = os.path.join(self.results_base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['plots', 'csv_files', 'tables', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)
            
        print(f"📁 Created run directory: {self.run_dir}")
        return self.run_dir

    def list_available_datasets(self):
        """List all available dataset folders"""
        print(f"📂 Available datasets in {self.base_datasets_dir}:")
        
        if not os.path.exists(self.base_datasets_dir):
            print(f"❌ Base datasets directory not found: {self.base_datasets_dir}")
            return []
        
        datasets = []
        for item in os.listdir(self.base_datasets_dir):
            item_path = os.path.join(self.base_datasets_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains subdirectories with HDF5 files
                has_test_data = self.check_dataset_has_test_data(item_path)
                status = "✅" if has_test_data else "⚠️"
                print(f"   {status} {item}")
                if has_test_data:
                    datasets.append(item)
        
        return datasets

    def check_dataset_has_test_data(self, dataset_path):
        """Check if dataset has test data in subdirectories"""
        try:
            for subdir in os.listdir(dataset_path):
                subdir_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(subdir_path):
                    test_files = list(Path(subdir_path).glob("*test.hdf5")) + \
                               list(Path(subdir_path).glob("*-dataset-test.hdf5"))
                    if test_files:
                        return True
            return False
        except:
            return False

    def check_folder_has_test_data(self, folder_path):
        """Check if folder has test HDF5 files"""
        test_files = list(Path(folder_path).glob("*test.hdf5")) + \
                    list(Path(folder_path).glob("*-dataset-test.hdf5"))
        return len(test_files) > 0

    def run_prediction_command(self, folder_path, dataset_name):
        """Run the prediction command and extract metrics"""
        print(f"\n🔍 Testing folder: {folder_path}")
        
        command = [
            "python3", "flatten_lucid/lucid_cnn_flatten.py",
            "--predict", folder_path,
            "--model", self.model_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Save command output to log file
            log_file = os.path.join(self.run_dir, 'logs', f'{dataset_name}_{os.path.basename(folder_path)}.log')
            with open(log_file, 'w') as f:
                f.write("Command: " + " ".join(command) + "\n\n")
                f.write("STDOUT:\n" + output + "\n\n")
                f.write("STDERR:\n" + result.stderr + "\n")
            
            # Parse the output to extract metrics
            metrics = self.parse_prediction_output(output)
            if metrics:
                metrics['dataset'] = dataset_name
                metrics['folder'] = os.path.basename(folder_path)
                metrics['full_path'] = folder_path
                return metrics
            else:
                print(f"❌ Could not parse metrics from output for {folder_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed for {folder_path}: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error for {folder_path}: {e}")
            return None

    def parse_prediction_output(self, output):
        """Parse the prediction output to extract metrics"""
        try:
            lines = output.strip().split('\n')
            
            # Find the start of the dictionary (line containing 'Model')
            dict_start_idx = -1
            for i, line in enumerate(lines):
                if "'Model':" in line:
                    dict_start_idx = i
                    break
            
            if dict_start_idx == -1:
                print(f"❌ Could not find metrics dictionary in output")
                return self.parse_metrics_manually(output)
            
            # Collect all lines from the dictionary start until we have a complete dict
            dict_lines = []
            for i in range(dict_start_idx, len(lines)):
                line = lines[i].strip()
                dict_lines.append(line)
                
                # Check if we have the complete dictionary (ends with 'Source': '...'}
                if "'Source':" in line and line.endswith("'}"):
                    break
            
            # Join the dictionary lines and clean them up
            dict_str = ' '.join(dict_lines)
            dict_str = dict_str.replace('\n', ' ').replace('\r', ' ')
            
            try:
                metrics_dict = eval(dict_str)
                
                return {
                    'accuracy': float(metrics_dict.get('Accuracy', 0)),
                    'f1_score': float(metrics_dict.get('F1Score', 0)),
                    'fpr': float(metrics_dict.get('FPR', 0)),
                    'fnr': float(metrics_dict.get('FNR', 0)),
                    'tpr': float(metrics_dict.get('TPR', 0)),
                    'tnr': float(metrics_dict.get('TNR', 0)),
                    'ddos_percentage': float(metrics_dict.get('DDOS%', 0)),
                    'processing_time': float(metrics_dict.get('Time', 0)),
                    'packets': int(metrics_dict.get('Packets', 0)),
                    'samples': int(metrics_dict.get('Samples', 0)),
                    'source_file': metrics_dict.get('Source', 'unknown')
                }
                
            except Exception as e:
                print(f"❌ Error evaluating dictionary: {e}")
                return self.parse_metrics_manually(output)
            
        except Exception as e:
            print(f"❌ Error parsing output: {e}")
            return None

    def parse_metrics_manually(self, output):
        """Manual parsing fallback for metrics"""
        try:
            patterns = {
                'accuracy': r"'Accuracy':\s*'([0-9.]+)'",
                'f1_score': r"'F1Score':\s*'([0-9.]+)'",
                'fpr': r"'FPR':\s*'([0-9.]+)'",
                'fnr': r"'FNR':\s*'([0-9.]+)'",
                'tpr': r"'TPR':\s*'([0-9.]+)'",
                'tnr': r"'TNR':\s*'([0-9.]+)'",
                'ddos_percentage': r"'DDOS%':\s*'([0-9.]+)'",
                'processing_time': r"'Time':\s*'([0-9.]+)'",
                'packets': r"'Packets':\s*([0-9]+)",
                'samples': r"'Samples':\s*([0-9]+)",
            }
            
            results = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    if key in ['packets', 'samples']:
                        results[key] = int(match.group(1))
                    else:
                        results[key] = float(match.group(1))
                else:
                    results[key] = 0
            
            source_match = re.search(r"'Source':\s*'([^']+)'", output)
            results['source_file'] = source_match.group(1) if source_match else 'unknown'
            
            return results
            
        except Exception as e:
            print(f"❌ Manual parsing failed: {e}")
            return None

    def test_datasets(self, dataset_names):
        """Test all specified datasets"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return []
        
        # Create run directory
        self.create_run_directory(dataset_names)
        
        all_results = []
        
        # Process each dataset
        for dataset_name in dataset_names:
            dataset_path = os.path.join(self.base_datasets_dir, dataset_name)
            
            print(f"\n{'='*80}")
            print(f"🚀 Processing dataset: {dataset_name}")
            print(f"📁 Path: {dataset_path}")
            print(f"{'='*80}")
            
            if not os.path.exists(dataset_path):
                print(f"❌ Dataset path does not exist: {dataset_path}")
                continue
            
            # Get all subdirectories
            subdirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
            subdirs.sort()
            
            print(f"📂 Found {len(subdirs)} subdirectories: {subdirs}")
            
            dataset_results = []
            
            for subdir in subdirs:
                folder_path = os.path.join(dataset_path, subdir)
                
                # Check if folder has test data
                if not self.check_folder_has_test_data(folder_path):
                    print(f"⚠️  Skipping {subdir}: No test HDF5 files found")
                    continue
                
                # Run prediction
                result = self.run_prediction_command(folder_path, dataset_name)
                if result:
                    dataset_results.append(result)
                    all_results.append(result)
                    print(f"✅ {subdir}: Acc={result['accuracy']:.3f}, F1={result['f1_score']:.3f}, FNR={result['fnr']:.3f}")
                else:
                    print(f"❌ Failed to get results for {subdir}")
            
            print(f"\n📊 {dataset_name} Summary:")
            print(f"   Successfully tested: {len(dataset_results)}/{len(subdirs)} folders")
            
            if dataset_results:
                avg_acc = np.mean([r['accuracy'] for r in dataset_results])
                avg_f1 = np.mean([r['f1_score'] for r in dataset_results])
                avg_fnr = np.mean([r['fnr'] for r in dataset_results])
                print(f"   Average Accuracy: {avg_acc:.3f}")
                print(f"   Average F1 Score: {avg_f1:.3f}")
                print(f"   Average FNR: {avg_fnr:.3f}")
        
        return all_results

    def save_results_to_csv(self, results):
        """Save results to CSV files"""
        if not results:
            print("❌ No results to save!")
            return None
        
        df = pd.DataFrame(results)
        
        # Main results CSV
        csv_filename = os.path.join(self.run_dir, 'csv_files', f'prediction_results_{self.timestamp}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Main results saved to: {csv_filename}")
        
        # Detailed tables
        self.create_detailed_tables(df)
        
        return csv_filename

    def create_detailed_tables(self, df):
        """Create detailed result tables"""
        df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
        
        tables_dir = os.path.join(self.run_dir, 'tables')
        
        # 1. Main results table
        results_table = df[['attack_type', 'dataset', 'accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr']].copy()
        results_table = results_table.sort_values(['attack_type', 'dataset'])
        results_table.to_csv(os.path.join(tables_dir, f'detailed_results_{self.timestamp}.csv'), index=False)
        
        # 2. Summary statistics by dataset
        summary_stats = df.groupby('dataset').agg({
            'accuracy': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'fnr': ['mean', 'std', 'min', 'max'],
            'fpr': ['mean', 'std'],
            'tpr': ['mean', 'std'],
            'tnr': ['mean', 'std']
        }).round(3)
        summary_stats.to_csv(os.path.join(tables_dir, f'summary_statistics_{self.timestamp}.csv'))
        
        # 3. Attack ranking by average FNR
        avg_metrics = df.groupby('attack_type').agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'fnr': ['mean', 'std', 'min', 'max']
        }).round(3)
        avg_metrics.columns = ['Avg_Acc', 'Avg_F1', 'Avg_FNR', 'Std_FNR', 'Min_FNR', 'Max_FNR']
        avg_metrics = avg_metrics.sort_values('Avg_FNR')
        avg_metrics.to_csv(os.path.join(tables_dir, f'attack_ranking_{self.timestamp}.csv'))
        
        # 4. Pivot tables for each metric
        self.create_pivot_tables(df)
        
        print(f"📄 Detailed tables saved in: {tables_dir}")

    def create_pivot_tables(self, df):
        """Create pivot tables for easy comparison"""
        df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
        
        metrics = {
            'Accuracy': 'accuracy',
            'F1_Score': 'f1_score',
            'False_Negative_Rate': 'fnr',
            'False_Positive_Rate': 'fpr',
            'True_Positive_Rate': 'tpr',
            'True_Negative_Rate': 'tnr'
        }
        
        pivot_file = os.path.join(self.run_dir, 'tables', f'pivot_tables_{self.timestamp}.csv')
        with open(pivot_file, 'w') as f:
            f.write("PIVOT TABLES FOR ALL METRICS\n")
            f.write("="*50 + "\n\n")
            
            for metric_name, metric_col in metrics.items():
                f.write(f"{metric_name} Pivot Table\n")
                f.write("-" * 30 + "\n")
                
                pivot = df.pivot(index='attack_type', columns='dataset', values=metric_col)
                pivot = pivot.round(3)
                
                # Add statistics columns
                pivot['Mean'] = pivot.mean(axis=1).round(3)
                pivot['Std'] = pivot.std(axis=1).round(3)
                pivot['Range'] = (pivot.max(axis=1) - pivot.min(axis=1)).round(3)
                
                pivot.to_csv(f, float_format='%.3f')
                f.write("\n\n")

    def generate_comparison_plots(self, df):
        """Generate comprehensive comparison plots"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        plots_dir = os.path.join(self.run_dir, 'plots')
        
        # Main comparison plot
        self.create_main_comparison_plot(df, plots_dir)
        
        # F1 Score and FNR focused plots
        self.create_f1_fnr_plots(df, plots_dir)
        
        # Detailed analysis plots
        self.create_detailed_analysis_plots(df, plots_dir)
        
        # Individual metric plots
        self.create_individual_metric_plots(df, plots_dir)

    def create_main_comparison_plot(self, df, plots_dir):
        """Create the main comparison plot with F1 and FNR focus"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        # 1. F1 Score comparison
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='dataset', y='f1_score', ax=ax1)
        sns.stripplot(data=df, x='dataset', y='f1_score', ax=ax1, color='black', alpha=0.6)
        ax1.set_title('F1 Score Comparison', fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_xlabel('Dataset')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. False Negative Rate comparison
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='dataset', y='fnr', ax=ax2)
        sns.stripplot(data=df, x='dataset', y='fnr', ax=ax2, color='black', alpha=0.6)
        ax2.set_title('False Negative Rate Comparison', fontweight='bold')
        ax2.set_ylabel('False Negative Rate')
        ax2.set_xlabel('Dataset')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Accuracy comparison
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='dataset', y='accuracy', ax=ax3)
        sns.stripplot(data=df, x='dataset', y='accuracy', ax=ax3, color='black', alpha=0.6)
        ax3.set_title('Accuracy Comparison', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Dataset')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Overall performance heatmap
        ax4 = axes[1, 1]
        avg_metrics = df.groupby('dataset')[['accuracy', 'f1_score', 'fnr']].mean()
        sns.heatmap(avg_metrics.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4)
        ax4.set_title('Average Metrics Heatmap', fontweight='bold')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Metrics')
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'main_comparison_{self.timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Main comparison plot saved: {plot_file}")

    def create_f1_fnr_plots(self, df, plots_dir):
        """Create focused F1 Score and FNR plots"""
        df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
        
        # F1 Score detailed plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # F1 Score by attack type and dataset
        attack_types = sorted(df['attack_type'].unique())
        datasets = sorted(df['dataset'].unique())
        
        x = np.arange(len(attack_types))
        width = 0.8 / len(datasets)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, dataset in enumerate(datasets):
            dataset_data = df[df['dataset'] == dataset]
            f1_values = []
            
            for attack in attack_types:
                attack_data = dataset_data[dataset_data['attack_type'] == attack]
                f1_values.append(attack_data['f1_score'].iloc[0] if len(attack_data) > 0 else 0)
            
            bars = ax1.bar(x + i*width, f1_values, width, label=dataset, 
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, f1_values):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax1.set_xlabel('Attack Types', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax1.set_title('F1 Score Comparison by Attack Type and Dataset', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax1.set_xticklabels(attack_types, rotation=45, ha='right')
        ax1.legend(title='Dataset', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.05)
        
        # F1 Score heatmap
        pivot_f1 = df.pivot(index='attack_type', columns='dataset', values='f1_score')
        pivot_f1 = pivot_f1.reindex(attack_types)
        
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='Greens', ax=ax2, 
                    cbar_kws={'label': 'F1 Score'}, linewidths=0.5, linecolor='white')
        ax2.set_title('F1 Score Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        f1_plot_file = os.path.join(plots_dir, f'f1_detailed_analysis_{self.timestamp}.png')
        plt.savefig(f1_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # FNR detailed plot (similar structure to original create_fn_comparison.py)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # FNR by attack type and dataset
        for i, dataset in enumerate(datasets):
            dataset_data = df[df['dataset'] == dataset]
            fnr_values = []
            
            for attack in attack_types:
                attack_data = dataset_data[dataset_data['attack_type'] == attack]
                fnr_values.append(attack_data['fnr'].iloc[0] if len(attack_data) > 0 else 0)
            
            bars = ax1.bar(x + i*width, fnr_values, width, label=dataset, 
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, fnr_values):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax1.set_xlabel('Attack Types', fontsize=12, fontweight='bold')
        ax1.set_ylabel('False Negative Rate', fontsize=12, fontweight='bold')
        ax1.set_title('False Negative Rate Comparison by Attack Type and Dataset', 
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax1.set_xticklabels(attack_types, rotation=45, ha='right')
        ax1.legend(title='Dataset', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.05)
        
        # FNR heatmap
        pivot_fnr = df.pivot(index='attack_type', columns='dataset', values='fnr')
        pivot_fnr = pivot_fnr.reindex(attack_types)
        
        sns.heatmap(pivot_fnr, annot=True, fmt='.3f', cmap='Reds', ax=ax2, 
                    cbar_kws={'label': 'False Negative Rate'}, linewidths=0.5, linecolor='white')
        ax2.set_title('False Negative Rate Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        fnr_plot_file = os.path.join(plots_dir, f'fnr_detailed_analysis_{self.timestamp}.png')
        plt.savefig(fnr_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 F1 Score detailed plot saved: {f1_plot_file}")
        print(f"📊 FNR detailed plot saved: {fnr_plot_file}")

    def create_detailed_analysis_plots(self, df, plots_dir):
        """Create detailed statistical analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. F1 vs FNR scatter plot
        ax1 = axes[0, 0]
        datasets = df['dataset'].unique()
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset'] == dataset]
            ax1.scatter(dataset_df['f1_score'], dataset_df['fnr'], 
                       label=dataset, alpha=0.7, s=60, color=colors[i])
        
        ax1.set_xlabel('F1 Score')
        ax1.set_ylabel('False Negative Rate')
        ax1.set_title('F1 Score vs FNR Correlation', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation line
        if len(df) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['f1_score'], df['fnr'])
            line = slope * df['f1_score'] + intercept
            ax1.plot(df['f1_score'], line, 'r--', alpha=0.8, 
                    label=f'Correlation: r={r_value:.3f}')
            ax1.legend()
        
        # 2. Performance distribution by dataset
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='dataset', y='f1_score', ax=ax2)
        ax2.set_title('F1 Score Distribution by Dataset', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Attack types ranked by average FNR
        ax3 = axes[1, 0]
        df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
        avg_fnr_by_attack = df.groupby('attack_type')['fnr'].mean().sort_values(ascending=False)
        
        bars = ax3.bar(range(len(avg_fnr_by_attack)), avg_fnr_by_attack.values, 
                       color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Attack Type (Ranked by avg FNR)')
        ax3.set_ylabel('Average False Negative Rate')
        ax3.set_title('Attack Types by Average FNR', fontweight='bold')
        ax3.set_xticks(range(len(avg_fnr_by_attack)))
        ax3.set_xticklabels(avg_fnr_by_attack.index, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Metrics correlation heatmap
        ax4 = axes[1, 1]
        corr_data = df[['accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr']].corr()
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Metrics Correlation Matrix', fontweight='bold')
        
        plt.tight_layout()
        analysis_plot_file = os.path.join(plots_dir, f'detailed_analysis_{self.timestamp}.png')
        plt.savefig(analysis_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Detailed analysis plot saved: {analysis_plot_file}")

    def create_individual_metric_plots(self, df, plots_dir):
        """Create individual plots for each major metric"""
        metrics = {
            'accuracy': 'Accuracy',
            'f1_score': 'F1 Score',
            'fnr': 'False Negative Rate',
            'fpr': 'False Positive Rate',
            'tpr': 'True Positive Rate',
            'tnr': 'True Negative Rate'
        }
        
        for metric_key, metric_name in metrics.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Box plot
            sns.boxplot(data=df, x='dataset', y=metric_key, ax=ax1)
            sns.stripplot(data=df, x='dataset', y=metric_key, ax=ax1, color='black', alpha=0.6)
            ax1.set_title(f'{metric_name} by Dataset', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Violin plot
            sns.violinplot(data=df, x='dataset', y=metric_key, ax=ax2)
            ax2.set_title(f'{metric_name} Distribution', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            metric_plot_file = os.path.join(plots_dir, f'{metric_key}_analysis_{self.timestamp}.png')
            plt.savefig(metric_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Individual metric plots saved in: {plots_dir}")

    def print_comprehensive_summary(self, df):
        """Print comprehensive summary statistics"""
        df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
        
        print(f"\n{'='*80}")
        print("📈 COMPREHENSIVE SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total folders tested: {len(df)}")
        print(f"   Datasets: {', '.join(df['dataset'].unique())}")
        print(f"   Attack types: {', '.join(sorted(df['attack_type'].unique()))}")
        
        print(f"\n🎯 Performance Metrics (Overall):")
        metrics = ['accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr']
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            print(f"   {metric.upper():<12}: {mean_val:.3f} ± {std_val:.3f} (min: {min_val:.3f}, max: {max_val:.3f})")
        
        print(f"\n📋 Performance by Dataset:")
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset]
            print(f"\n   {dataset}:")
            print(f"      Folders tested: {len(dataset_df)}")
            for metric in ['accuracy', 'f1_score', 'fnr']:
                mean_val = dataset_df[metric].mean()
                std_val = dataset_df[metric].std()
                print(f"      {metric:<12}: {mean_val:.3f} ± {std_val:.3f}")
        
        print(f"\n🏆 Best and Worst Performers:")
        
        # Best performers
        best_acc = df.loc[df['accuracy'].idxmax()]
        print(f"   Best Accuracy: {best_acc['attack_type']} ({best_acc['dataset']}) - {best_acc['accuracy']:.3f}")
        
        best_f1 = df.loc[df['f1_score'].idxmax()]
        print(f"   Best F1 Score: {best_f1['attack_type']} ({best_f1['dataset']}) - {best_f1['f1_score']:.3f}")
        
        best_fnr = df.loc[df['fnr'].idxmin()]
        print(f"   Lowest FNR: {best_fnr['attack_type']} ({best_fnr['dataset']}) - {best_fnr['fnr']:.3f}")
        
        # Worst performers
        worst_acc = df.loc[df['accuracy'].idxmin()]
        print(f"   Worst Accuracy: {worst_acc['attack_type']} ({worst_acc['dataset']}) - {worst_acc['accuracy']:.3f}")
        
        worst_f1 = df.loc[df['f1_score'].idxmin()]
        print(f"   Worst F1 Score: {worst_f1['attack_type']} ({worst_f1['dataset']}) - {worst_f1['f1_score']:.3f}")
        
        highest_fnr = df.loc[df['fnr'].idxmax()]
        print(f"   Highest FNR: {highest_fnr['attack_type']} ({highest_fnr['dataset']}) - {highest_fnr['fnr']:.3f}")

        # Save summary to file
        summary_file = os.path.join(self.run_dir, f'summary_report_{self.timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write("UNIFIED TEST AND ANALYSIS SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Used: {self.model_path}\n")
            f.write(f"Datasets Tested: {', '.join(df['dataset'].unique())}\n")
            f.write(f"Total Tests: {len(df)}\n\n")
            
            # Add all the statistics
            f.write("OVERALL STATISTICS:\n")
            for metric in metrics:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                f.write(f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}\n")
            
        print(f"\n💾 Summary report saved: {summary_file}")

    def run_complete_analysis(self, dataset_names):
        """Run complete testing and analysis pipeline"""
        print(f"🚀 Starting unified testing and analysis for datasets: {', '.join(dataset_names)}")
        
        # Test datasets
        results = self.test_datasets(dataset_names)
        
        if not results:
            print("❌ No results obtained from any dataset!")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results to CSV
        csv_file = self.save_results_to_csv(results)
        
        # Generate all plots
        self.generate_comparison_plots(df)
        
        # Print comprehensive summary
        self.print_comprehensive_summary(df)
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 All results saved in: {self.run_dir}")
        print(f"   📊 Plots: {os.path.join(self.run_dir, 'plots')}")
        print(f"   📄 CSV files: {os.path.join(self.run_dir, 'csv_files')}")
        print(f"   📋 Tables: {os.path.join(self.run_dir, 'tables')}")
        print(f"   📝 Logs: {os.path.join(self.run_dir, 'logs')}")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Unified Testing and Analysis Script for LUCID DDoS Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 unified_test_and_analyze.py MANIPULATED-FLATTEN_0 MANIPULATED-FLATTEN_42
  python3 unified_test_and_analyze.py FLATTEN-PCAPS
  python3 unified_test_and_analyze.py --list-datasets
  python3 unified_test_and_analyze.py --help
        """)
    
    parser.add_argument('datasets', nargs='*', 
                       help='Names of dataset folders to test (space-separated)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available dataset folders')
    parser.add_argument('--base-dir', default='./TrafficManipulator-master/DATASETS',
                       help='Base directory containing datasets (default: ./TrafficManipulator-master/DATASETS)')
    parser.add_argument('--model', default='./output/10t-100n-DOS2019-LUCID-FLATTEN.h5',
                       help='Path to the model file (default: ./output/10t-100n-DOS2019-LUCID-FLATTEN.h5)')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = UnifiedTester(base_datasets_dir=args.base_dir, model_path=args.model)
    
    # List datasets if requested
    if args.list_datasets:
        available = tester.list_available_datasets()
        print(f"\n📋 Available datasets with test data: {len(available)}")
        return
    
    # Check if datasets were provided
    if not args.datasets:
        print("❌ No datasets specified!")
        print("Use --list-datasets to see available datasets")
        print("Or specify dataset names: python3 unified_test_and_analyze.py DATASET1 DATASET2")
        return
    
    # Validate datasets exist
    available_datasets = tester.list_available_datasets()
    invalid_datasets = [d for d in args.datasets if d not in available_datasets]
    
    if invalid_datasets:
        print(f"❌ Invalid datasets specified: {', '.join(invalid_datasets)}")
        print(f"Available datasets: {', '.join(available_datasets)}")
        return
    
    # Run the complete analysis
    success = tester.run_complete_analysis(args.datasets)
    
    if success:
        print(f"\n🎉 Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
