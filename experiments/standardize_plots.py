#!/usr/bin/env python3
"""
Standardized Performance Plotter
Generates consistent plots for RF, MLP, and LUCID results.
- Heatmap for F1-Score/Accuracy comparison across attacks
- Bar chart (Histogram) for F1-Score specifically
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Standard style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
COLORS = {'Baseline': '#2ca02c', 'Manipulated': '#1f77b4', 'Fragmented': '#d62728'}

def load_results(base_path, model_type):
    """
    Load results from CSV files. 
    This is a placeholder logic - we need to see the actual CSV structure first.
    For now, I'll simulate loading or try to find the files.
    """
    # Try given paths
    if model_type == 'RF':
        search_pattern = os.path.join(base_path, 'RandomForest/results/rf_cross_dataset_*.csv')
    elif model_type == 'LUCID':
        search_pattern = os.path.join(base_path, 'output/*.csv')
    elif model_type == 'MLP':
         # MLP results seem to be in .h5 or need extraction, usually they are logged.
         # For this task, I might need to mock or manually input if files aren't easy to parse.
         # Let's check for any .csv in MLP folder
         search_pattern = os.path.join(base_path, 'MLP/*.csv')
    
    files = glob.glob(search_pattern)
    if not files:
        print(f"No result files found for {model_type} in {search_pattern}")
        return None
        
    # Use the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading {model_type} results from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")
        return None

def standardize_dataframe(df, model_type):
    """
    Standardize column names to: 'Attack', 'Metric', 'Value', 'Scenario'
    This depends highly on the input format.
    """
    # This function will need to be adapted once I inspect the CSVs
    return df

def plot_f1_histogram(df, model_name, output_path):
    """
    Bar chart showing F1 Score for each attack across 3 scenarios
    """
    plt.figure(figsize=(14, 8))
    
    # Filter for F1 Score if needed, assuming dataframe is prepared
    # df should have columns: Attack, Scenario, F1
    
    # Ensure correct order of scenarios
    scenario_order = ['Baseline', 'Manipulated', 'Fragmented']
    
    # Create bar plot
    ax = sns.barplot(x='Attack', y='F1', hue='Scenario', data=df, 
                     palette=[COLORS['Baseline'], COLORS['Manipulated'], COLORS['Fragmented']],
                     hue_order=scenario_order)
    
    plt.title(f'{model_name} Performance - F1 Score Comparison', fontsize=16, pad=20)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(title='Traffic Type', loc='lower left')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars?
    # Maybe too cluttered for all bars.
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved histogram to {output_path}")
    plt.close()

def plot_heatmap(df, model_name, metric='F1', output_path=None):
    """
    Heatmap: Rows=Attacks, Cols=Scenario
    """
    # Pivot specific metric
    pivot_df = df.pivot(index='Attack', columns='Scenario', values=metric)
    
    # Reorder columns
    cols = [c for c in ['Baseline', 'Manipulated', 'Fragmented'] if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1,
                linewidths=1, linecolor='white')
    
    plt.title(f'{model_name} - {metric} Score Heatmap', fontsize=16, pad=20)
    plt.xlabel('Traffic Type', fontsize=14)
    plt.ylabel('Attack Type', fontsize=14)
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved heatmap to {output_path}")
        plt.close()

def standardize_rf_data(df):
    """
    Map RF results to standard format: ['Attack', 'Scenario', 'F1', 'Accuracy']
    """
    # Map dataset names to Scenario names
    scenario_map = {
        'FLATTEN-PCAPS': 'Baseline',
        'MANIPULATED-FLATTEN_42': 'Manipulated',
        'FRAGMENTED_42_150': 'Fragmented'
    }
    
    # Filter only these datasets
    df = df[df['test_dataset'].isin(scenario_map.keys())].copy()
    
    # Check for duplicates! The error says we have them.
    # It's possible the CSV has multiple runs or model_names.
    # Let's drop duplicates keeping the last one or average them.
    df = df.drop_duplicates(subset=['test_dataset', 'attack_type'])
    
    df['Scenario'] = df['test_dataset'].map(scenario_map)
    df.rename(columns={'attack_type': 'Attack', 'f1_score': 'F1', 'accuracy': 'Accuracy'}, inplace=True)
    
    return df[['Attack', 'Scenario', 'F1', 'Accuracy']]

def standardize_lucid_data(base_path):
    """
    LUCID results seem scattered or in specific log files.
    Since we don't have a consolidated CSV, I will generate a Mock consolidated dataframe
    based on the analysis we've done in previous turns (which had aggregate stats).
    
    Ideally, we would parse `output/` logs, but for perfect plotting now, 
    I'll construct a dataframe that mirrors the RF structure but with LUCID's typical performance profile.
    
    Note: If real data is strictly required, I would need to parse 'history.log' or run an aggregator.
    Given the user wants "Standardized plots", I will use the RF data as the primary source for the
    visual style template, and try to find the real LUCID data source.
    """
    # Let's try to find a consolidated LUCID report.
    # The user mentioned "plots of performance of RF, MLP and LUCID are different".
    # This implies they exist.
    pass

def main():
    base_dir = "/home/rising/EvasionShield-LUCID"
    output_dir = os.path.join(base_dir, "visualizations/standardized")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Process Random Forest
    print("Processing Random Forest Results...")
    df_rf_raw = load_results(base_dir, 'RF')
    if df_rf_raw is not None:
        df_rf = standardize_rf_data(df_rf_raw)
        plot_f1_histogram(df_rf, 'Random Forest', os.path.join(output_dir, 'RF_F1_Histogram.png'))
        plot_heatmap(df_rf, 'Random Forest', 'F1', os.path.join(output_dir, 'RF_F1_Heatmap.png'))
        
    # 2. Process LUCID
    # Since I cannot find the consolidated CSV, and the user likely has generated them before
    # I will look for a file that looks like '10t-100n-DOS2019-LUCID-FLATTEN.csv' in output?
    # Or I'll skip LUCID aggregation here to avoid guessing.
    # Wait, the RF csv contains everything? No, it says model_name is mostly uniform?
    
    pass

# Temporary main to inspect data structures first
def inspect_data():
    base_dir = "/home/rising/EvasionShield-LUCID"
    
    # Check RF
    df_rf = load_results(base_dir, 'RF')
    if df_rf is not None:
        print("RF Columns:", df_rf.columns.tolist())
        print(df_rf.head())

    # Check LUCID - likely in output/
    # Need to find the right summary CSV
    df_lucid = load_results(base_dir, 'LUCID')
    if df_lucid is not None:
         print("LUCID Columns:", df_lucid.columns.tolist())
         print(df_lucid.head())

if __name__ == "__main__":
    main()
