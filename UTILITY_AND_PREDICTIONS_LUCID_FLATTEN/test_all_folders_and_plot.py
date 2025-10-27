#!/usr/bin/env python3
"""
Script to test all folders in MANIPULATED-FLATTEN_0, MANIPULATED_FLATTEN_42, and FLATTEN-PCAPS
and generate comparison plots for F1 score, accuracy, and false negative rate.
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

def run_prediction_command(folder_path, model_path, dataset_name):
    """Run the prediction command and extract metrics"""
    print(f"\n🔍 Testing folder: {folder_path}")
    
    command = [
        "python3", "flatten_lucid/lucid_cnn_flatten.py",
        "--predict", folder_path,
        "--model", model_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the output to extract metrics
        metrics = parse_prediction_output(output)
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

def parse_prediction_output(output):
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
            return None
        
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
        
        # Remove any extra whitespace and ensure proper formatting
        dict_str = dict_str.replace('\n', ' ').replace('\r', ' ')
        
        # Use eval to parse the dictionary (safer alternatives could be ast.literal_eval)
        try:
            metrics_dict = eval(dict_str)
            
            return {
                'accuracy': float(metrics_dict.get('Accuracy', 0)),
                'f1_score': float(metrics_dict.get('F1Score', 0)),
                'fpr': float(metrics_dict.get('FPR', 0)),  # False Positive Rate
                'fnr': float(metrics_dict.get('FNR', 0)),  # False Negative Rate
                'tpr': float(metrics_dict.get('TPR', 0)),  # True Positive Rate (Sensitivity)
                'tnr': float(metrics_dict.get('TNR', 0)),  # True Negative Rate (Specificity)
                'ddos_percentage': float(metrics_dict.get('DDOS%', 0)),
                'processing_time': float(metrics_dict.get('Time', 0)),
                'packets': int(metrics_dict.get('Packets', 0)),
                'samples': int(metrics_dict.get('Samples', 0)),
                'source_file': metrics_dict.get('Source', 'unknown')
            }
            
        except Exception as e:
            print(f"❌ Error evaluating dictionary: {e}")
            print(f"Dict string was: {dict_str}")
            return parse_metrics_manually(output)
        
    except Exception as e:
        print(f"❌ Error parsing output: {e}")
        return None

def parse_metrics_manually(output):
    """Manual parsing fallback for metrics"""
    import re
    
    try:
        # Use regex to extract metrics
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
        
        # Extract source file
        source_match = re.search(r"'Source':\s*'([^']+)'", output)
        results['source_file'] = source_match.group(1) if source_match else 'unknown'
        
        return results
        
    except Exception as e:
        print(f"❌ Manual parsing failed: {e}")
        return None

def check_folder_has_test_data(folder_path):
    """Check if folder has test HDF5 files"""
    test_files = list(Path(folder_path).glob("*test.hdf5")) + list(Path(folder_path).glob("*-dataset-test.hdf5"))
    return len(test_files) > 0

def main():
    # Configuration
    base_datasets_dir = "./TrafficManipulator-master/DATASETS"
    model_path = "./output/10t-100n-DOS2019-LUCID-FLATTEN.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Define the datasets to test
    datasets_info = [
        {
            'name': 'MANIPULATED-FLATTEN_0',
            'path': os.path.join(base_datasets_dir, 'MANIPULATED-FLATTEN_0'),
            'description': 'Original manipulated dataset'
        },
        {
            'name': 'MANIPULATED-FLATTEN_12345',
            'path': os.path.join(base_datasets_dir, 'MANIPULATED-FLATTEN_12345'),
            'description': 'Manipulated dataset variant 12345'
        },
        {
            'name': 'MANIPULATED-FLATTEN_42',
            'path': os.path.join(base_datasets_dir, 'MANIPULATED-FLATTEN_42'),
            'description': 'Manipulated dataset variant 42'
        },
        {
            'name': 'FLATTEN-PCAPS',
            'path': os.path.join(base_datasets_dir, 'FLATTEN-PCAPS'),
            'description': 'Flattened PCAP dataset'
        }
    ]
    
    all_results = []
    
    # Process each dataset
    for dataset_info in datasets_info:
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        
        print(f"\n{'='*80}")
        print(f"🚀 Processing dataset: {dataset_name}")
        print(f"📁 Path: {dataset_path}")
        print(f"📝 Description: {dataset_info['description']}")
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
            if not check_folder_has_test_data(folder_path):
                print(f"⚠️  Skipping {subdir}: No test HDF5 files found")
                continue
            
            # Run prediction
            result = run_prediction_command(folder_path, model_path, dataset_name)
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
    
    if not all_results:
        print("❌ No results obtained from any dataset!")
        sys.exit(1)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"prediction_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to: {csv_filename}")
    
    # Generate plots
    generate_comparison_plots(df, timestamp)
    
    # Print summary statistics
    print_summary_statistics(df)

def generate_comparison_plots(df, timestamp):
    """Generate comparison plots for the metrics"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='dataset', y='accuracy', ax=ax1)
    sns.stripplot(data=df, x='dataset', y='accuracy', ax=ax1, color='black', alpha=0.6)
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Dataset')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='dataset', y='f1_score', ax=ax2)
    sns.stripplot(data=df, x='dataset', y='f1_score', ax=ax2, color='black', alpha=0.6)
    ax2.set_title('F1 Score Comparison', fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_xlabel('Dataset')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. False Negative Rate comparison
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='dataset', y='fnr', ax=ax3)
    sns.stripplot(data=df, x='dataset', y='fnr', ax=ax3, color='black', alpha=0.6)
    ax3.set_title('False Negative Rate Comparison', fontweight='bold')
    ax3.set_ylabel('False Negative Rate')
    ax3.set_xlabel('Dataset')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Overall performance heatmap
    ax4 = axes[1, 1]
    
    # Calculate average metrics per dataset
    avg_metrics = df.groupby('dataset')[['accuracy', 'f1_score', 'fnr']].mean()
    
    # Create heatmap
    sns.heatmap(avg_metrics.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4)
    ax4.set_title('Average Metrics Heatmap', fontweight='bold')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Metrics')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"performance_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_filename}")
    
    # Create individual detailed plots
    create_detailed_plots(df, timestamp)
    
    plt.show()

def create_detailed_plots(df, timestamp):
    """Create detailed individual plots"""
    
    # 1. Per-folder performance plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create a grouped bar plot
    folders = df['folder'].unique()
    datasets = df['dataset'].unique()
    
    x = np.arange(len(folders))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        accuracies = [dataset_data[dataset_data['folder'] == folder]['accuracy'].iloc[0] 
                     if len(dataset_data[dataset_data['folder'] == folder]) > 0 else 0
                     for folder in folders]
        
        ax.bar(x + i*width, accuracies, width, label=dataset, alpha=0.8)
    
    ax.set_xlabel('Attack Type Folders')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Attack Type and Dataset')
    ax.set_xticks(x + width)
    ax.set_xticklabels(folders, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"accuracy_by_folder_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"📊 Detailed accuracy plot saved to: accuracy_by_folder_{timestamp}.png")
    
    # 2. Correlation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create correlation matrix
    corr_data = df[['accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr', 'processing_time']].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Metrics Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(f"metrics_correlation_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"📊 Correlation plot saved to: metrics_correlation_{timestamp}.png")

def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    
    print(f"\n{'='*80}")
    print("📈 COMPREHENSIVE SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total folders tested: {len(df)}")
    print(f"   Datasets: {', '.join(df['dataset'].unique())}")
    print(f"   Attack types: {', '.join(sorted(df['folder'].unique()))}")
    
    print(f"\n🎯 Performance Metrics (Overall):")
    metrics = ['accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr']
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"   {metric.upper():<12}: {mean_val:.3f} ± {std_val:.3f} (min: {min_val:.3f}, max: {max_val:.3f})")
    
    print(f"\n📋 Performance by Dataset:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        print(f"\n   {dataset}:")
        print(f"      Folders tested: {len(dataset_df)}")
        for metric in ['accuracy', 'f1_score', 'fnr']:
            mean_val = dataset_df[metric].mean()
            std_val = dataset_df[metric].std()
            print(f"      {metric:<12}: {mean_val:.3f} ± {std_val:.3f}")
    
    print(f"\n🏆 Best and Worst Performers:")
    
    # Best accuracy
    best_acc = df.loc[df['accuracy'].idxmax()]
    print(f"   Best Accuracy: {best_acc['folder']} ({best_acc['dataset']}) - {best_acc['accuracy']:.3f}")
    
    # Best F1 score
    best_f1 = df.loc[df['f1_score'].idxmax()]
    print(f"   Best F1 Score: {best_f1['folder']} ({best_f1['dataset']}) - {best_f1['f1_score']:.3f}")
    
    # Lowest FNR (best)
    best_fnr = df.loc[df['fnr'].idxmin()]
    print(f"   Lowest FNR: {best_fnr['folder']} ({best_fnr['dataset']}) - {best_fnr['fnr']:.3f}")
    
    # Worst performers
    worst_acc = df.loc[df['accuracy'].idxmin()]
    print(f"   Worst Accuracy: {worst_acc['folder']} ({worst_acc['dataset']}) - {worst_acc['accuracy']:.3f}")
    
    worst_f1 = df.loc[df['f1_score'].idxmin()]
    print(f"   Worst F1 Score: {worst_f1['folder']} ({worst_f1['dataset']}) - {worst_f1['f1_score']:.3f}")
    
    highest_fnr = df.loc[df['fnr'].idxmax()]
    print(f"   Highest FNR: {highest_fnr['folder']} ({highest_fnr['dataset']}) - {highest_fnr['fnr']:.3f}")

if __name__ == "__main__":
    main()
