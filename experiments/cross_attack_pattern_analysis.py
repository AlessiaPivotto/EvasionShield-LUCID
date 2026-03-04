#!/usr/bin/env python3
"""
Cross-Attack Feature Importance Pattern Analysis
Analyzes how feature importance patterns vary across different DDoS attack types
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

def create_feature_mapping():
    """Create comprehensive mapping of LUCID network flow features"""
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
    
    # Feature categories for pattern analysis
    feature_categories = {
        'packet_counts': [1, 2],
        'packet_sizes': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'timing_features': [0, 15, 16, 17, 18, 19, 20],
        'flow_rates': [13, 14]
    }
    
    return feature_map, feature_categories

def load_attack_dataset(attack_path, attack_name, max_samples=2000):
    """Load and preprocess attack dataset"""
    print(f"\nLoading {attack_name} attack data from: {attack_path}")
    
    try:
        with h5py.File(attack_path, 'r') as f:
            X = f['set_x'][:]
            y = f['set_y'][:]
            
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        print(f"  Label distribution: {np.bincount(y)}")
        
        # Sample if too large
        if max_samples and X.shape[0] > max_samples:
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"  Sampled to {max_samples} instances")
        
        return X, y
        
    except Exception as e:
        print(f"  Error loading {attack_name}: {e}")
        return None, None

def analyze_single_attack_importance(X, y, attack_name):
    """Analyze feature importance for a single attack type"""
    print(f"\n--- Analyzing {attack_name} Attack ---")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get baseline accuracy
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Model accuracy: {accuracy:.4f}")
    
    # Feature importance
    importance_scores = rf.feature_importances_
    
    # Permutation importance for validation
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
    perm_scores = perm_importance.importances_mean
    
    # Individual feature impact analysis
    feature_impacts = []
    baseline_pred = rf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    for feature_idx in range(X.shape[1]):
        X_test_reset = X_test.copy()
        X_test_reset[:, feature_idx] = 0
        
        reset_pred = rf.predict(X_test_reset)
        reset_acc = accuracy_score(y_test, reset_pred)
        impact = baseline_acc - reset_acc
        
        feature_impacts.append(impact)
    
    return {
        'attack_name': attack_name,
        'accuracy': accuracy,
        'rf_importance': importance_scores,
        'perm_importance': perm_scores,
        'feature_impacts': np.array(feature_impacts),
        'model': rf
    }

def compare_feature_patterns(attack_results, feature_map, feature_categories):
    """Compare feature importance patterns across attack types"""
    print(f"\n{'='*60}")
    print("CROSS-ATTACK FEATURE IMPORTANCE PATTERNS")
    print(f"{'='*60}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for result in attack_results:
        attack_name = result['attack_name']
        for feature_idx in range(len(result['rf_importance'])):
            comparison_data.append({
                'attack_type': attack_name,
                'feature_idx': feature_idx,
                'feature_name': feature_map.get(feature_idx, f"Feature {feature_idx}"),
                'rf_importance': result['rf_importance'][feature_idx],
                'perm_importance': result['perm_importance'][feature_idx],
                'feature_impact': result['feature_impacts'][feature_idx],
                'accuracy': result['accuracy']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Analysis 1: Top features by attack type
    print("\n1. TOP 5 MOST IMPORTANT FEATURES BY ATTACK TYPE")
    print("-" * 55)
    
    for attack in df['attack_type'].unique():
        attack_data = df[df['attack_type'] == attack].sort_values('rf_importance', ascending=False).head(5)
        print(f"\n{attack.upper()}:")
        for _, row in attack_data.iterrows():
            print(f"  {row['feature_idx']:2d}. {row['feature_name']:<25} | "
                  f"Importance: {row['rf_importance']:.4f} | "
                  f"Impact: {row['feature_impact']:.4f}")
    
    # Analysis 2: Feature category importance
    print(f"\n\n2. FEATURE CATEGORY ANALYSIS")
    print("-" * 35)
    
    category_analysis = {}
    for category, feature_indices in feature_categories.items():
        category_analysis[category] = {}
        
        for attack in df['attack_type'].unique():
            attack_data = df[df['attack_type'] == attack]
            category_importance = attack_data[attack_data['feature_idx'].isin(feature_indices)]['rf_importance'].sum()
            category_impact = attack_data[attack_data['feature_idx'].isin(feature_indices)]['feature_impact'].sum()
            
            category_analysis[category][attack] = {
                'importance': category_importance,
                'impact': category_impact
            }
    
    for category, attacks in category_analysis.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for attack, metrics in attacks.items():
            print(f"  {attack:>12}: Importance={metrics['importance']:.4f}, Impact={metrics['impact']:.4f}")
    
    # Analysis 3: Attack vulnerability patterns
    print(f"\n\n3. ATTACK VULNERABILITY PATTERNS")
    print("-" * 40)
    
    vulnerability_summary = {}
    for attack in df['attack_type'].unique():
        attack_data = df[df['attack_type'] == attack]
        
        # Find most vulnerable features (highest impact)
        most_vulnerable = attack_data.nlargest(3, 'feature_impact')
        total_vulnerability = attack_data['feature_impact'].sum()
        max_single_impact = attack_data['feature_impact'].max()
        
        vulnerability_summary[attack] = {
            'total_vulnerability': total_vulnerability,
            'max_single_impact': max_single_impact,
            'most_vulnerable_features': most_vulnerable[['feature_idx', 'feature_name', 'feature_impact']].to_dict('records')
        }
        
        print(f"\n{attack.upper()}:")
        print(f"  Total vulnerability: {total_vulnerability:.4f}")
        print(f"  Max single feature impact: {max_single_impact:.4f}")
        print(f"  Most vulnerable features:")
        for vuln_feat in vulnerability_summary[attack]['most_vulnerable_features']:
            print(f"    {vuln_feat['feature_idx']:2d}. {vuln_feat['feature_name']:<25} | Impact: {vuln_feat['feature_impact']:.4f}")
    
    return df, category_analysis, vulnerability_summary

def create_pattern_visualizations(df, category_analysis, vulnerability_summary, feature_map):
    """Create comprehensive visualizations of feature importance patterns"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Feature importance heatmap
    plt.subplot(3, 3, 1)
    importance_pivot = df.pivot(index='feature_name', columns='attack_type', values='rf_importance')
    sns.heatmap(importance_pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar=True)
    plt.title('Feature Importance Across Attack Types', fontsize=12, fontweight='bold')
    plt.ylabel('Features')
    plt.xlabel('Attack Types')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Plot 2: Feature impact heatmap
    plt.subplot(3, 3, 2)
    impact_pivot = df.pivot(index='feature_name', columns='attack_type', values='feature_impact')
    sns.heatmap(impact_pivot, annot=True, fmt='.3f', cmap='Reds', cbar=True)
    plt.title('Feature Impact When Removed', fontsize=12, fontweight='bold')
    plt.ylabel('Features')
    plt.xlabel('Attack Types')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Plot 3: Category importance comparison
    plt.subplot(3, 3, 3)
    category_data = []
    for category, attacks in category_analysis.items():
        for attack, metrics in attacks.items():
            category_data.append({
                'category': category.replace('_', ' ').title(),
                'attack': attack,
                'importance': metrics['importance']
            })
    
    cat_df = pd.DataFrame(category_data)
    cat_pivot = cat_df.pivot(index='category', columns='attack', values='importance')
    sns.heatmap(cat_pivot, annot=True, fmt='.3f', cmap='Blues', cbar=True)
    plt.title('Feature Category Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Categories')
    plt.xlabel('Attack Types')
    plt.xticks(rotation=45)
    
    # Plot 4: Top features by attack (bar chart)
    plt.subplot(3, 3, 4)
    attack_types = df['attack_type'].unique()
    top_features_data = []
    
    for attack in attack_types:
        attack_data = df[df['attack_type'] == attack].nlargest(5, 'rf_importance')
        for _, row in attack_data.iterrows():
            top_features_data.append({
                'attack': attack,
                'feature': f"F{row['feature_idx']}",
                'importance': row['rf_importance']
            })
    
    top_df = pd.DataFrame(top_features_data)
    
    # Create grouped bar chart
    attack_positions = np.arange(len(attack_types))
    bar_width = 0.15
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i in range(5):  # Top 5 features
        feature_values = []
        for attack in attack_types:
            attack_features = top_df[top_df['attack'] == attack]
            if len(attack_features) > i:
                feature_values.append(attack_features.iloc[i]['importance'])
            else:
                feature_values.append(0)
        
        plt.bar(attack_positions + i * bar_width, feature_values, 
               bar_width, label=f'Feature {i+1}', color=colors[i], alpha=0.8)
    
    plt.xlabel('Attack Types')
    plt.ylabel('Feature Importance')
    plt.title('Top 5 Features by Attack Type', fontsize=12, fontweight='bold')
    plt.xticks(attack_positions + bar_width * 2, attack_types, rotation=45)
    plt.legend()
    
    # Plot 5: Vulnerability comparison
    plt.subplot(3, 3, 5)
    vuln_attacks = list(vulnerability_summary.keys())
    total_vulns = [vulnerability_summary[attack]['total_vulnerability'] for attack in vuln_attacks]
    max_impacts = [vulnerability_summary[attack]['max_single_impact'] for attack in vuln_attacks]
    
    x = np.arange(len(vuln_attacks))
    width = 0.35
    
    plt.bar(x - width/2, total_vulns, width, label='Total Vulnerability', color='lightcoral', alpha=0.8)
    plt.bar(x + width/2, max_impacts, width, label='Max Single Impact', color='lightblue', alpha=0.8)
    
    plt.xlabel('Attack Types')
    plt.ylabel('Vulnerability Score')
    plt.title('Attack Vulnerability Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, vuln_attacks, rotation=45)
    plt.legend()
    
    # Plot 6: Feature correlation across attacks
    plt.subplot(3, 3, 6)
    correlation_matrix = importance_pivot.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
               cmap='RdBu_r', center=0, cbar=True)
    plt.title('Attack Pattern Correlations', fontsize=12, fontweight='bold')
    
    # Plot 7: Feature ranking stability
    plt.subplot(3, 3, 7)
    feature_rankings = {}
    for attack in attack_types:
        attack_data = df[df['attack_type'] == attack].sort_values('rf_importance', ascending=False)
        feature_rankings[attack] = attack_data['feature_idx'].tolist()[:10]  # Top 10
    
    # Calculate ranking consistency
    all_features = set()
    for rankings in feature_rankings.values():
        all_features.update(rankings[:5])  # Top 5 from each
    
    consistency_scores = {}
    for feature in all_features:
        positions = []
        for attack, rankings in feature_rankings.items():
            if feature in rankings:
                positions.append(rankings.index(feature) + 1)
            else:
                positions.append(11)  # Not in top 10
        
        consistency_scores[feature] = np.std(positions)
    
    # Plot feature stability
    features = list(consistency_scores.keys())
    stabilities = list(consistency_scores.values())
    feature_names = [feature_map.get(f, f"F{f}") for f in features]
    
    plt.barh(range(len(features)), stabilities, color='lightgreen', alpha=0.7)
    plt.yticks(range(len(features)), [f"F{f}" for f in features])
    plt.xlabel('Ranking Instability (lower = more stable)')
    plt.title('Feature Ranking Stability', fontsize=12, fontweight='bold')
    
    # Plot 8: Attack distinguishability
    plt.subplot(3, 3, 8)
    
    # Calculate how well each feature distinguishes between attacks
    distinguishability = {}
    for feature_idx in range(21):  # Assuming 21 features
        feature_data = df[df['feature_idx'] == feature_idx]
        if len(feature_data) == len(attack_types):
            importance_values = feature_data['rf_importance'].values
            distinguishability[feature_idx] = np.std(importance_values)
    
    sorted_features = sorted(distinguishability.items(), key=lambda x: x[1], reverse=True)[:10]
    
    features_dist = [f[0] for f in sorted_features]
    dist_scores = [f[1] for f in sorted_features]
    feature_names_dist = [feature_map.get(f, f"F{f}") for f in features_dist]
    
    plt.bar(range(len(features_dist)), dist_scores, color='orange', alpha=0.7)
    plt.xticks(range(len(features_dist)), [f"F{f}" for f in features_dist], rotation=45)
    plt.ylabel('Distinguishability Score')
    plt.title('Most Attack-Distinguishing Features', fontsize=12, fontweight='bold')
    
    # Plot 9: Feature category breakdown by attack
    plt.subplot(3, 3, 9)
    
    # Stack plot of category contributions
    category_contributions = {}
    for attack in attack_types:
        category_contributions[attack] = []
        for category in category_analysis.keys():
            category_contributions[attack].append(category_analysis[category][attack]['importance'])
    
    bottom = np.zeros(len(attack_types))
    colors_cat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, category in enumerate(category_analysis.keys()):
        values = [category_contributions[attack][i] for attack in attack_types]
        plt.bar(attack_types, values, bottom=bottom, label=category.replace('_', ' ').title(), 
               color=colors_cat[i % len(colors_cat)], alpha=0.8)
        bottom += values
    
    plt.xlabel('Attack Types')
    plt.ylabel('Cumulative Importance')
    plt.title('Feature Category Contributions', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = '/home/rising/EvasionShield-LUCID/experiments/cross_attack_feature_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive visualization saved to: {output_path}")
    
    plt.show()

def generate_strategic_insights(df, category_analysis, vulnerability_summary):
    """Generate strategic insights for defense improvements"""
    
    print(f"\n{'='*80}")
    print("STRATEGIC INSIGHTS FOR DEFENSE SYSTEM")
    print(f"{'='*80}")
    
    # Find universal critical features
    universal_features = {}
    for feature_idx in df['feature_idx'].unique():
        feature_data = df[df['feature_idx'] == feature_idx]
        avg_importance = feature_data['rf_importance'].mean()
        avg_impact = feature_data['feature_impact'].mean()
        importance_std = feature_data['rf_importance'].std()
        
        if avg_importance > 0.05:  # Significant features only
            universal_features[feature_idx] = {
                'avg_importance': avg_importance,
                'avg_impact': avg_impact,
                'consistency': 1 / (importance_std + 0.001)  # Higher = more consistent
            }
    
    # Sort by average importance
    sorted_universal = sorted(universal_features.items(), 
                             key=lambda x: x[1]['avg_importance'], reverse=True)[:5]
    
    print("\n1. UNIVERSAL CRITICAL FEATURES (Target for Protection)")
    print("-" * 55)
    feature_map, _ = create_feature_mapping()
    for feature_idx, metrics in sorted_universal:
        feature_name = feature_map.get(feature_idx, f"Feature {feature_idx}")
        print(f"  Feature {feature_idx:2d}: {feature_name:<25} | "
              f"Avg Importance: {metrics['avg_importance']:.4f} | "
              f"Avg Impact: {metrics['avg_impact']:.4f}")
    
    # Attack-specific vulnerabilities
    print(f"\n2. ATTACK-SPECIFIC VULNERABILITIES")
    print("-" * 40)
    
    for attack, vuln_data in vulnerability_summary.items():
        print(f"\n{attack.upper()} - Primary Weakness:")
        most_vulnerable = vuln_data['most_vulnerable_features'][0]
        feature_name = feature_map.get(most_vulnerable['feature_idx'], f"Feature {most_vulnerable['feature_idx']}")
        print(f"  Feature {most_vulnerable['feature_idx']:2d}: {feature_name} "
              f"(Impact: {most_vulnerable['feature_impact']:.4f})")
    
    # Defense recommendations
    print(f"\n3. DEFENSE RECOMMENDATIONS")
    print("-" * 30)
    
    # Category-based recommendations
    print(f"\nA. FEATURE CATEGORY PRIORITIES:")
    category_priorities = {}
    for category, attacks in category_analysis.items():
        total_importance = sum([attack_data['importance'] for attack_data in attacks.values()])
        total_impact = sum([attack_data['impact'] for attack_data in attacks.values()])
        category_priorities[category] = total_importance + total_impact
    
    sorted_categories = sorted(category_priorities.items(), key=lambda x: x[1], reverse=True)
    
    for i, (category, priority_score) in enumerate(sorted_categories, 1):
        print(f"  {i}. {category.replace('_', ' ').title():<20} | Priority Score: {priority_score:.4f}")
    
    print(f"\nB. ATTACK-SPECIFIC COUNTERMEASURES:")
    
    # Fragmented attack (highest single vulnerability)
    fragmented_vuln = vulnerability_summary.get('fragmented', {})
    if fragmented_vuln:
        max_impact = fragmented_vuln.get('max_single_impact', 0)
        if max_impact > 0.3:  # High impact threshold
            print(f"  FRAGMENTED ATTACKS - CRITICAL:")
            print(f"    • Implement redundant timing measurements")
            print(f"    • Add backup features for Flow IAT calculations") 
            print(f"    • Consider packet reassembly preprocessing")
    
    # Manipulated attack (different pattern)
    manipulated_vuln = vulnerability_summary.get('manipulated', {})
    if manipulated_vuln:
        print(f"  MANIPULATED ATTACKS:")
        print(f"    • Focus on packet count validation")
        print(f"    • Implement flow duration cross-checks")
        print(f"    • Add feature integrity monitoring")
    
    # Baseline (clean traffic patterns)
    baseline_vuln = vulnerability_summary.get('baseline', {})
    if baseline_vuln:
        print(f"  BASELINE TRAFFIC:")
        print(f"    • Ensure robust feature extraction")
        print(f"    • Implement quality checks for timing features")
        print(f"    • Add anomaly detection for feature corruption")
    
    print(f"\nC. IMPLEMENTATION PRIORITIES:")
    print(f"  1. HIGH: Protect universal critical features first")
    print(f"  2. MEDIUM: Implement attack-specific defenses")
    print(f"  3. LOW: Optimize less critical feature categories")
    
    # Create summary table
    summary_data = []
    for attack in vulnerability_summary.keys():
        attack_data = df[df['attack_type'] == attack]
        top_feature = attack_data.nlargest(1, 'rf_importance').iloc[0]
        
        summary_data.append({
            'Attack Type': attack.title(),
            'Most Important Feature': f"F{top_feature['feature_idx']} ({top_feature['feature_name']})",
            'Importance Score': f"{top_feature['rf_importance']:.4f}",
            'Max Vulnerability': f"{vulnerability_summary[attack]['max_single_impact']:.4f}",
            'Total Risk Score': f"{vulnerability_summary[attack]['total_vulnerability']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n4. ATTACK SUMMARY TABLE")
    print("-" * 25)
    print(summary_df.to_string(index=False))
    
    # Save strategic summary
    summary_path = '/home/rising/EvasionShield-LUCID/experiments/strategic_defense_insights.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nStrategic insights saved to: {summary_path}")

def main():
    """Main analysis function"""
    
    # Define attack datasets
    attack_datasets = {
        'baseline': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'manipulated': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42/09-TFTP/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'fragmented': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FRAGMENTED_42_150/09-TFTP/10t-100n-DOS2019-flatten-dataset-test.hdf5'
    }
    
    print("="*80)
    print("CROSS-ATTACK FEATURE IMPORTANCE PATTERN ANALYSIS")
    print("="*80)
    print("Analyzing feature importance patterns across different DDoS attack types...")
    
    # Get feature mappings
    feature_map, feature_categories = create_feature_mapping()
    
    # Analyze each attack type
    attack_results = []
    
    for attack_name, attack_path in attack_datasets.items():
        if not os.path.exists(attack_path):
            print(f"Warning: {attack_name} dataset not found: {attack_path}")
            continue
            
        X, y = load_attack_dataset(attack_path, attack_name)
        if X is not None:
            result = analyze_single_attack_importance(X, y, attack_name)
            attack_results.append(result)
    
    if not attack_results:
        print("Error: No attack datasets could be loaded!")
        return
    
    # Cross-attack pattern analysis
    df, category_analysis, vulnerability_summary = compare_feature_patterns(
        attack_results, feature_map, feature_categories
    )
    
    # Create visualizations
    create_pattern_visualizations(df, category_analysis, vulnerability_summary, feature_map)
    
    # Generate strategic insights
    generate_strategic_insights(df, category_analysis, vulnerability_summary)
    
    # Save detailed results
    detailed_path = '/home/rising/EvasionShield-LUCID/experiments/cross_attack_analysis_detailed.csv'
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed analysis results saved to: {detailed_path}")
    
    print(f"\n{'='*80}")
    print("CROSS-ATTACK ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Key outputs generated:")
    print("  • Comprehensive feature pattern analysis")
    print("  • Visual comparison across attack types")
    print("  • Strategic defense recommendations")
    print("  • Detailed CSV results for further analysis")

if __name__ == "__main__":
    main()
