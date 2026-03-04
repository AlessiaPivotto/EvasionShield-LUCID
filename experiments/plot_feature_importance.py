#!/usr/bin/env python3
"""
Enhanced Feature Importance Visualization for LUCID
Creates clear, publication-ready plots of feature importance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_enhanced_importance_plots():
    """Create enhanced feature importance visualizations"""
    
    # Load the analysis results
    df = pd.read_csv('/home/rising/EvasionShield-LUCID/experiments/feature_importance_analysis.csv')
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top Features by Dataset (Bar Plot)
    plt.subplot(2, 3, 1)
    
    # Get top 5 features per dataset
    top_features_data = []
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset].head(5)
        for _, row in dataset_data.iterrows():
            top_features_data.append(row)
    
    top_df = pd.DataFrame(top_features_data)
    
    # Create grouped bar plot
    datasets = df['dataset'].unique()
    feature_names = df['feature_name'].unique()[:8]  # Top 8 features
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, dataset in enumerate(datasets):
        dataset_scores = []
        for fname in feature_names:
            score = df[(df['dataset'] == dataset) & (df['feature_name'] == fname)]['importance_score']
            dataset_scores.append(score.iloc[0] if len(score) > 0 else 0)
        
        plt.bar(x + i*width, dataset_scores, width, label=dataset.title(), 
               color=colors[i], alpha=0.8)
    
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Feature Importance by Dataset', fontsize=14, fontweight='bold')
    plt.xticks(x + width, [name.replace(' ', '\n') for name in feature_names], rotation=0, fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Critical Feature Impact (Heatmap)
    plt.subplot(2, 3, 2)
    
    # Create pivot table for relative drops
    pivot_impact = df.pivot_table(index='feature_name', columns='dataset', 
                                 values='relative_drop', fill_value=0)
    
    # Select top features by maximum impact
    max_impacts = pivot_impact.max(axis=1).sort_values(ascending=False)
    top_impact_features = max_impacts.head(10).index
    pivot_subset = pivot_impact.loc[top_impact_features]
    
    sns.heatmap(pivot_subset, annot=True, fmt='.1f', cmap='Reds', 
                cbar_kws={'label': 'Accuracy Drop (%)'}, linewidths=0.5)
    plt.title('Feature Impact Heatmap\n(Accuracy Drop %)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 3. Top 10 Most Critical Features (Horizontal Bar)
    plt.subplot(2, 3, 3)
    
    # Get the features with highest impact across all datasets
    max_impact_per_feature = df.groupby('feature_name')['relative_drop'].max().sort_values(ascending=True)
    top_critical = max_impact_per_feature.tail(10)
    
    colors_bar = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_critical)))
    bars = plt.barh(range(len(top_critical)), top_critical.values, color=colors_bar)
    
    plt.yticks(range(len(top_critical)), top_critical.index, fontsize=10)
    plt.xlabel('Maximum Impact (% Accuracy Drop)', fontsize=12, fontweight='bold')
    plt.title('Most Critical Features\n(Maximum Impact)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=10)
    
    # 4. Feature Ranking Consistency
    plt.subplot(2, 3, 4)
    
    # Show how consistently features rank high across datasets
    rank_data = []
    for feature in df['feature_name'].unique():
        feature_data = df[df['feature_name'] == feature]
        avg_rank = feature_data['rank'].mean()
        rank_data.append({'feature': feature, 'avg_rank': avg_rank, 'datasets': len(feature_data)})
    
    rank_df = pd.DataFrame(rank_data)
    rank_df = rank_df[rank_df['datasets'] >= 2]  # Features in at least 2 datasets
    rank_df = rank_df.sort_values('avg_rank').head(10)
    
    plt.scatter(rank_df['avg_rank'], range(len(rank_df)), 
               s=rank_df['datasets']*100, alpha=0.7, c='#FF6B6B')
    
    plt.yticks(range(len(rank_df)), rank_df['feature'])
    plt.xlabel('Average Rank', fontsize=12, fontweight='bold')
    plt.title('Feature Ranking Consistency\n(Size = # Datasets)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Dataset Vulnerability Profile
    plt.subplot(2, 3, 5)
    
    # Show vulnerability of each dataset to feature removal
    vulnerability_data = []
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        max_impact = dataset_data['relative_drop'].max()
        avg_impact = dataset_data['relative_drop'].mean()
        num_critical = len(dataset_data[dataset_data['relative_drop'] > 5])  # >5% impact
        
        vulnerability_data.append({
            'dataset': dataset,
            'max_impact': max_impact,
            'avg_impact': avg_impact,
            'critical_features': num_critical
        })
    
    vuln_df = pd.DataFrame(vulnerability_data)
    
    x_pos = np.arange(len(vuln_df))
    
    plt.bar(x_pos, vuln_df['max_impact'], alpha=0.7, label='Max Impact', color='#FF6B6B')
    plt.bar(x_pos, vuln_df['avg_impact'], alpha=0.7, label='Avg Impact', color='#4ECDC4')
    
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Impact (% Accuracy Drop)', fontsize=12, fontweight='bold')
    plt.title('Dataset Vulnerability Profile', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, vuln_df['dataset'].str.title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add critical feature count as text
    for i, (max_val, crit_count) in enumerate(zip(vuln_df['max_impact'], vuln_df['critical_features'])):
        plt.text(i, max_val + 1, f'{int(crit_count)} critical\nfeatures', 
                ha='center', va='bottom', fontsize=9)
    
    # 6. Feature Type Analysis
    plt.subplot(2, 3, 6)
    
    # Categorize features by type
    def categorize_feature(name):
        if 'IAT' in name or 'Duration' in name:
            return 'Timing'
        elif 'Packet' in name:
            return 'Packet Count'
        elif 'Length' in name:
            return 'Size/Length'
        elif 'Flow' in name:
            return 'Flow Stats'
        else:
            return 'Other'
    
    df['feature_category'] = df['feature_name'].apply(categorize_feature)
    
    # Calculate average impact by category
    category_impact = df.groupby('feature_category')['relative_drop'].agg(['mean', 'max', 'count'])
    category_impact = category_impact.sort_values('max', ascending=False)
    
    # Create stacked bar chart
    x_pos = np.arange(len(category_impact))
    plt.bar(x_pos, category_impact['max'], alpha=0.7, label='Max Impact', color='#FF6B6B')
    plt.bar(x_pos, category_impact['mean'], alpha=0.7, label='Avg Impact', color='#4ECDC4')
    
    plt.xlabel('Feature Category', fontsize=12, fontweight='bold')
    plt.ylabel('Impact (% Accuracy Drop)', fontsize=12, fontweight='bold')
    plt.title('Feature Category Vulnerability', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, category_impact.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add feature count as text
    for i, count in enumerate(category_impact['count']):
        plt.text(i, category_impact.iloc[i]['max'] + 0.5, f'({int(count)})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(pad=3.0)
    
    # Save the enhanced plot
    output_path = '/home/rising/EvasionShield-LUCID/experiments/enhanced_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Enhanced feature importance plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Create summary statistics
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE SUMMARY STATISTICS")
    print("="*80)
    
    # Most critical feature overall
    most_critical = df.loc[df['relative_drop'].idxmax()]
    print(f"\nMost Critical Feature:")
    print(f"  {most_critical['feature_name']} (Feature {most_critical['feature_idx']})")
    print(f"  Dataset: {most_critical['dataset'].title()}")
    print(f"  Impact: {most_critical['relative_drop']:.1f}% accuracy drop")
    print(f"  Importance Score: {most_critical['importance_score']:.4f}")
    
    # Most consistently important features
    consistency_scores = df.groupby('feature_name').agg({
        'rank': 'mean',
        'relative_drop': 'mean',
        'importance_score': 'mean',
        'dataset': 'count'
    }).sort_values('rank')
    
    print(f"\nMost Consistently Important Features (across multiple datasets):")
    consistent_features = consistency_scores[consistency_scores['dataset'] >= 2].head(5)
    for idx, (feature, data) in enumerate(consistent_features.iterrows(), 1):
        print(f"  {idx}. {feature}")
        print(f"     Avg Rank: {data['rank']:.1f}")
        print(f"     Avg Impact: {data['relative_drop']:.1f}%")
        print(f"     Present in {int(data['dataset'])} datasets")
    
    # Category analysis
    print(f"\nFeature Category Vulnerability Ranking:")
    category_summary = df.groupby('feature_category').agg({
        'relative_drop': ['mean', 'max'],
        'feature_name': 'count'
    }).round(2)
    
    category_summary.columns = ['Avg_Impact', 'Max_Impact', 'Count']
    category_summary = category_summary.sort_values('Max_Impact', ascending=False)
    
    for idx, (category, data) in enumerate(category_summary.iterrows(), 1):
        print(f"  {idx}. {category}: Max {data['Max_Impact']:.1f}%, Avg {data['Avg_Impact']:.1f}% ({int(data['Count'])} features)")

if __name__ == "__main__":
    print("Creating enhanced feature importance visualizations...")
    create_enhanced_importance_plots()
