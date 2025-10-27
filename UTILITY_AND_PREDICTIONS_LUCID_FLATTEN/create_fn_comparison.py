#!/usr/bin/env python3
"""
Script to create a detailed False Negative comparison plot for each attack type
across all three datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def create_fn_comparison_plot(csv_file):
    """Create detailed False Negative comparison plots"""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Set2")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean up folder names for better display
    df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
    
    # Sort attack types for consistent ordering
    attack_order = sorted(df['attack_type'].unique())
    
    # Create the main FN comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Grouped bar chart of FN rates by attack type
    attack_types = df['attack_type'].unique()
    datasets = df['dataset'].unique()
    
    x = np.arange(len(attack_types))
    width = 0.2  # Reduced width to accommodate 4 datasets
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        fn_values = []
        
        for attack in attack_types:
            attack_data = dataset_data[dataset_data['attack_type'] == attack]
            if len(attack_data) > 0:
                fn_values.append(attack_data['fnr'].iloc[0])
            else:
                fn_values.append(0)
        
        bars = ax1.bar(x + i*width, fn_values, width, label=dataset, 
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, fn_values)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax1.set_xlabel('Attack Types', fontsize=12, fontweight='bold')
    ax1.set_ylabel('False Negative Rate', fontsize=12, fontweight='bold')
    ax1.set_title('False Negative Rate Comparison by Attack Type and Dataset', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)  # Center the labels for 4 datasets
    ax1.set_xticklabels(attack_types, rotation=45, ha='right')
    ax1.legend(title='Dataset', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Heatmap of FN rates
    # Pivot the data for heatmap
    pivot_data = df.pivot(index='attack_type', columns='dataset', values='fnr')
    pivot_data = pivot_data.reindex(attack_order)
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='Reds', 
                ax=ax2, cbar_kws={'label': 'False Negative Rate'},
                linewidths=0.5, linecolor='white')
    ax2.set_title('False Negative Rate Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"fn_comparison_detailed_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 False Negative comparison plot saved to: {plot_filename}")
    
    # Create additional detailed plots
    create_fn_statistics_plot(df, timestamp)
    
    plt.show()
    
    return plot_filename

def create_fn_statistics_plot(df, timestamp):
    """Create additional FN statistics and analysis plots"""
    
    # Create a figure with multiple subplots for detailed FN analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed False Negative Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plot of FN rates by dataset
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='dataset', y='fnr', ax=ax1)
    sns.stripplot(data=df, x='dataset', y='fnr', ax=ax1, color='red', alpha=0.7, size=6)
    ax1.set_title('FN Rate Distribution by Dataset', fontweight='bold')
    ax1.set_ylabel('False Negative Rate')
    ax1.set_xlabel('Dataset')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: FN rate vs Accuracy scatter plot
    ax2 = axes[0, 1]
    datasets = df['dataset'].unique()
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset]
        ax2.scatter(dataset_df['accuracy'], dataset_df['fnr'], 
                   label=dataset, alpha=0.7, s=60, color=colors[i])
    
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('False Negative Rate')
    ax2.set_title('FN Rate vs Accuracy Correlation', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['accuracy'], df['fnr'])
    line = slope * df['accuracy'] + intercept
    ax2.plot(df['accuracy'], line, 'r--', alpha=0.8, 
            label=f'Correlation: r={r_value:.3f}')
    ax2.legend()
    
    # Plot 3: Attack types ranked by average FN rate
    ax3 = axes[1, 0]
    avg_fn_by_attack = df.groupby('attack_type')['fnr'].mean().sort_values(ascending=False)
    
    bars = ax3.bar(range(len(avg_fn_by_attack)), avg_fn_by_attack.values, 
                   color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Attack Type (Ranked by avg FN rate)')
    ax3.set_ylabel('Average False Negative Rate')
    ax3.set_title('Attack Types Ranked by Average FN Rate', fontweight='bold')
    ax3.set_xticks(range(len(avg_fn_by_attack)))
    ax3.set_xticklabels(avg_fn_by_attack.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_fn_by_attack.values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: FN rate variability across datasets
    ax4 = axes[1, 1]
    fn_std_by_attack = df.groupby('attack_type')['fnr'].std().sort_values(ascending=False)
    
    bars = ax4.bar(range(len(fn_std_by_attack)), fn_std_by_attack.values, 
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Attack Type (Ranked by FN variability)')
    ax4.set_ylabel('FN Rate Standard Deviation')
    ax4.set_title('FN Rate Variability Across Datasets', fontweight='bold')
    ax4.set_xticks(range(len(fn_std_by_attack)))
    ax4.set_xticklabels(fn_std_by_attack.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, fn_std_by_attack.values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the statistics plot
    stats_filename = f"fn_statistics_analysis_{timestamp}.png"
    plt.savefig(stats_filename, dpi=300, bbox_inches='tight')
    print(f"📊 FN statistics analysis saved to: {stats_filename}")
    
    return stats_filename

def print_fn_analysis(csv_file):
    """Print detailed FN analysis statistics"""
    
    df = pd.read_csv(csv_file)
    df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
    
    print(f"\n{'='*80}")
    print("🎯 DETAILED FALSE NEGATIVE ANALYSIS")
    print(f"{'='*80}")
    
    # Overall FN statistics
    print(f"\n📊 Overall FN Statistics:")
    print(f"   Mean FN Rate: {df['fnr'].mean():.3f}")
    print(f"   Median FN Rate: {df['fnr'].median():.3f}")
    print(f"   Min FN Rate: {df['fnr'].min():.3f}")
    print(f"   Max FN Rate: {df['fnr'].max():.3f}")
    print(f"   Std Deviation: {df['fnr'].std():.3f}")
    
    # FN by dataset
    print(f"\n📋 FN Rate by Dataset:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        print(f"   {dataset}:")
        print(f"      Mean: {dataset_df['fnr'].mean():.3f}")
        print(f"      Std:  {dataset_df['fnr'].std():.3f}")
        print(f"      Range: [{dataset_df['fnr'].min():.3f}, {dataset_df['fnr'].max():.3f}]")
    
    # Best and worst attacks by FN rate
    print(f"\n🏆 Attack Performance Analysis:")
    
    # Best performing attacks (lowest avg FN)
    avg_fn_by_attack = df.groupby('attack_type')['fnr'].mean().sort_values()
    print(f"\n   ✅ Best Attacks (Lowest Avg FN Rate):")
    for i, (attack, fn_rate) in enumerate(avg_fn_by_attack.head(5).items()):
        print(f"      {i+1}. {attack}: {fn_rate:.3f}")
    
    # Worst performing attacks (highest avg FN)
    print(f"\n   ❌ Worst Attacks (Highest Avg FN Rate):")
    for i, (attack, fn_rate) in enumerate(avg_fn_by_attack.tail(5).items()):
        print(f"      {i+1}. {attack}: {fn_rate:.3f}")
    
    # Most variable attacks
    fn_std_by_attack = df.groupby('attack_type')['fnr'].std().sort_values(ascending=False)
    print(f"\n   📈 Most Variable Attacks (Across Datasets):")
    for i, (attack, std_dev) in enumerate(fn_std_by_attack.head(5).items()):
        print(f"      {i+1}. {attack}: σ={std_dev:.3f}")
    
    # Dataset-specific analysis
    print(f"\n📈 Dataset-Specific FN Analysis:")
    pivot_data = df.pivot(index='attack_type', columns='dataset', values='fnr')
    
    for attack in pivot_data.index:
        values = pivot_data.loc[attack]
        min_fn = values.min()
        max_fn = values.max()
        range_fn = max_fn - min_fn
        
        if range_fn > 0.3:  # Significant variation
            best_dataset = values.idxmin()
            worst_dataset = values.idxmax()
            print(f"   🎯 {attack}:")
            print(f"      Best:  {best_dataset} (FN={min_fn:.3f})")
            print(f"      Worst: {worst_dataset} (FN={max_fn:.3f})")
            print(f"      Range: {range_fn:.3f}")

def create_results_table(csv_file):
    """Create formatted tables with all the results"""
    
    df = pd.read_csv(csv_file)
    df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comprehensive results table
    print(f"\n{'='*120}")
    print("📊 COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*120}")
    
    # Create pivot tables for each metric
    metrics = {
        'Accuracy': 'accuracy',
        'F1 Score': 'f1_score', 
        'False Negative Rate': 'fnr',
        'False Positive Rate': 'fpr',
        'True Positive Rate': 'tpr',
        'True Negative Rate': 'tnr'
    }
    
    # Sort attack types for consistent display
    attack_order = sorted(df['attack_type'].unique())
    
    # Main results table
    print(f"\n📋 Main Performance Metrics Table:")
    print(f"{'Attack Type':<15} {'Dataset':<20} {'Accuracy':<10} {'F1 Score':<10} {'FNR':<8} {'FPR':<8} {'TPR':<8} {'TNR':<8}")
    print("-" * 120)
    
    for attack in attack_order:
        attack_data = df[df['attack_type'] == attack].sort_values('dataset')
        for idx, row in attack_data.iterrows():
            attack_name = attack if idx == attack_data.index[0] else ""
            print(f"{attack_name:<15} {row['dataset']:<20} {row['accuracy']:<10.3f} {row['f1_score']:<10.3f} {row['fnr']:<8.3f} {row['fpr']:<8.3f} {row['tpr']:<8.3f} {row['tnr']:<8.3f}")
        print("-" * 120)
    
    # Summary statistics table
    print(f"\n📊 Summary Statistics by Dataset:")
    print(f"{'Dataset':<20} {'Avg Accuracy':<12} {'Avg F1':<10} {'Avg FNR':<10} {'Std FNR':<10} {'Min FNR':<10} {'Max FNR':<10}")
    print("-" * 100)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        print(f"{dataset:<20} {dataset_df['accuracy'].mean():<12.3f} {dataset_df['f1_score'].mean():<10.3f} "
              f"{dataset_df['fnr'].mean():<10.3f} {dataset_df['fnr'].std():<10.3f} "
              f"{dataset_df['fnr'].min():<10.3f} {dataset_df['fnr'].max():<10.3f}")
    
    # Attack type ranking table
    print(f"\n🏆 Attack Type Performance Ranking (by Average FNR):")
    avg_metrics = df.groupby('attack_type').agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'fnr': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    avg_metrics.columns = ['Avg_Acc', 'Avg_F1', 'Avg_FNR', 'Std_FNR', 'Min_FNR', 'Max_FNR']
    avg_metrics = avg_metrics.sort_values('Avg_FNR')
    
    print(f"{'Rank':<5} {'Attack Type':<15} {'Avg Acc':<9} {'Avg F1':<8} {'Avg FNR':<9} {'Std FNR':<9} {'Range FNR':<12}")
    print("-" * 80)
    
    for rank, (attack, row) in enumerate(avg_metrics.iterrows(), 1):
        range_fnr = f"{row['Min_FNR']:.3f}-{row['Max_FNR']:.3f}"
        performance_indicator = "🟢" if row['Avg_FNR'] < 0.1 else "🟡" if row['Avg_FNR'] < 0.3 else "🔴"
        print(f"{rank:<5} {attack:<15} {row['Avg_Acc']:<9.3f} {row['Avg_F1']:<8.3f} "
              f"{row['Avg_FNR']:<9.3f} {row['Std_FNR']:<9.3f} {range_fnr:<12} {performance_indicator}")
    
    # Save tables to files
    # 1. Main results table
    results_table = df[['attack_type', 'dataset', 'accuracy', 'f1_score', 'fnr', 'fpr', 'tpr', 'tnr']].copy()
    results_table = results_table.sort_values(['attack_type', 'dataset'])
    results_table.to_csv(f'detailed_results_table_{timestamp}.csv', index=False)
    
    # 2. Summary statistics table
    summary_stats = df.groupby('dataset').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'fnr': ['mean', 'std', 'min', 'max'],
        'fpr': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'tnr': ['mean', 'std']
    }).round(3)
    summary_stats.to_csv(f'summary_statistics_{timestamp}.csv')
    
    # 3. Attack ranking table
    avg_metrics.to_csv(f'attack_ranking_{timestamp}.csv')
    
    # 4. Pivot table for each metric
    create_pivot_tables(df, timestamp)
    
    print(f"\n💾 Tables saved:")
    print(f"   📄 detailed_results_table_{timestamp}.csv")
    print(f"   📄 summary_statistics_{timestamp}.csv") 
    print(f"   📄 attack_ranking_{timestamp}.csv")
    print(f"   📄 pivot_tables_{timestamp}.csv")
    
    return timestamp

def create_pivot_tables(df, timestamp):
    """Create pivot tables for easy comparison"""
    
    # Clean attack type names
    df['attack_type'] = df['folder'].str.replace('00-WebDDos', '00-WebDDoS')
    
    # Create pivot tables for each metric
    metrics = {
        'Accuracy': 'accuracy',
        'F1_Score': 'f1_score',
        'False_Negative_Rate': 'fnr',
        'False_Positive_Rate': 'fpr',
        'True_Positive_Rate': 'tpr',
        'True_Negative_Rate': 'tnr'
    }
    
    with open(f'pivot_tables_{timestamp}.csv', 'w') as f:
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
    
    print(f"\n📊 Pivot Tables Created:")
    for metric_name in metrics.keys():
        pivot = df.pivot(index='attack_type', columns='dataset', values=metrics[metric_name])
        print(f"\n{metric_name}:")
        print(pivot.round(3).to_string())

def main():
    csv_file = "prediction_results_20251023_185031.csv"
    
    print("🚀 Creating detailed False Negative comparison plots...")
    
    # Create the plots
    plot_file = create_fn_comparison_plot(csv_file)
    
    # Print detailed analysis
    print_fn_analysis(csv_file)
    
    # Create and display tables
    timestamp = create_results_table(csv_file)
    
    print(f"\n✅ Analysis complete! Check the generated plot files and CSV tables.")

if __name__ == "__main__":
    main()
