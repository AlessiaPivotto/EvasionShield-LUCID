#!/usr/bin/env python3
"""
LUCID Feature Analysis Tool
===========================

This script provides targeted feature analysis for LUCID DDoS detection models,
focusing on understanding which packet-level and statistical features are most
important for detection accuracy.

For LUCID CNN models, features represent:
- Rows: Packets in temporal order within time window
- Columns: Packet features (IAT, packet length, flags, etc.)

For LUCID Flatten models, features represent:
- Statistical aggregations (mean, std, min, max) of packet features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

class LucidFeatureAnalyzer:
    """Analyzes feature importance in LUCID models using feature reset technique."""
    
    def __init__(self):
        self.feature_descriptions = self._get_lucid_feature_descriptions()
    
    def _get_lucid_feature_descriptions(self):
        """Get descriptions of LUCID features based on the original paper."""
        return {
            'cnn_features': {
                0: 'Inter-arrival time (IAT)',
                1: 'Packet length',
                2: 'TCP Window size',
                3: 'TCP Flags',
                4: 'Protocol type',
                5: 'Normalized timestamp'
            },
            'flatten_features': {
                'iat_stats': 'Inter-arrival time statistics (mean, std, min, max)',
                'length_stats': 'Packet length statistics',
                'window_stats': 'TCP window size statistics',
                'flag_stats': 'TCP flags statistics',
                'protocol_stats': 'Protocol distribution statistics',
                'temporal_stats': 'Temporal flow characteristics'
            }
        }
    
    def analyze_cnn_features(self, model, X_test, Y_test, max_packets=None, max_features=None):
        """
        Analyze CNN model features by zeroing individual packet/feature combinations.
        
        Args:
            model: Trained LUCID CNN model
            X_test: Test data (samples, packets, features, channels)
            Y_test: Test labels
            max_packets: Maximum number of packets to analyze (for speed)
            max_features: Maximum number of features to analyze (for speed)
        """
        print("🔍 Analyzing CNN features...")
        
        # Get baseline performance
        baseline_pred = model.predict(X_test, batch_size=256, verbose=0)
        baseline_acc = self._calculate_accuracy(Y_test, baseline_pred)
        
        height, width = X_test.shape[1:3]
        max_packets = max_packets or height
        max_features = max_features or width
        
        results = []
        
        # Analyze individual packet/feature combinations
        for packet_idx in range(min(max_packets, height)):
            for feature_idx in range(min(max_features, width)):
                print(f"   Packet {packet_idx}, Feature {feature_idx} ({self.feature_descriptions['cnn_features'].get(feature_idx, 'Unknown')})")
                
                # Zero out specific packet/feature
                X_modified = X_test.copy()
                X_modified[:, packet_idx, feature_idx] = 0
                
                # Evaluate
                modified_pred = model.predict(X_modified, batch_size=256, verbose=0)
                modified_acc = self._calculate_accuracy(Y_test, modified_pred)
                
                accuracy_drop = baseline_acc - modified_acc
                
                results.append({
                    'packet_idx': packet_idx,
                    'feature_idx': feature_idx,
                    'feature_name': self.feature_descriptions['cnn_features'].get(feature_idx, 'Unknown'),
                    'position': f'P{packet_idx}_F{feature_idx}',
                    'baseline_accuracy': baseline_acc,
                    'modified_accuracy': modified_acc,
                    'accuracy_drop': accuracy_drop,
                    'relative_importance': accuracy_drop / baseline_acc if baseline_acc > 0 else 0
                })
        
        return pd.DataFrame(results)
    
    def analyze_packet_importance(self, model, X_test, Y_test):
        """Analyze importance of entire packets (all features of specific packets)."""
        print("🔍 Analyzing packet-level importance...")
        
        baseline_pred = model.predict(X_test, batch_size=256, verbose=0)
        baseline_acc = self._calculate_accuracy(Y_test, baseline_pred)
        
        height = X_test.shape[1]
        results = []
        
        for packet_idx in range(height):
            print(f"   Analyzing packet {packet_idx+1}/{height}")
            
            # Zero out entire packet
            X_modified = X_test.copy()
            X_modified[:, packet_idx, :] = 0
            
            # Evaluate
            modified_pred = model.predict(X_modified, batch_size=256, verbose=0)
            modified_acc = self._calculate_accuracy(Y_test, modified_pred)
            
            accuracy_drop = baseline_acc - modified_acc
            
            results.append({
                'packet_idx': packet_idx,
                'packet_position': f'Packet_{packet_idx}',
                'baseline_accuracy': baseline_acc,
                'modified_accuracy': modified_acc,
                'accuracy_drop': accuracy_drop,
                'relative_importance': accuracy_drop / baseline_acc if baseline_acc > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def analyze_feature_type_importance(self, model, X_test, Y_test):
        """Analyze importance of feature types across all packets."""
        print("🔍 Analyzing feature type importance...")
        
        baseline_pred = model.predict(X_test, batch_size=256, verbose=0)
        baseline_acc = self._calculate_accuracy(Y_test, baseline_pred)
        
        width = X_test.shape[2]
        results = []
        
        for feature_idx in range(width):
            feature_name = self.feature_descriptions['cnn_features'].get(feature_idx, 'Unknown')
            print(f"   Analyzing feature type: {feature_name}")
            
            # Zero out this feature across all packets
            X_modified = X_test.copy()
            X_modified[:, :, feature_idx] = 0
            
            # Evaluate
            modified_pred = model.predict(X_modified, batch_size=256, verbose=0)
            modified_acc = self._calculate_accuracy(Y_test, modified_pred)
            
            accuracy_drop = baseline_acc - modified_acc
            
            results.append({
                'feature_idx': feature_idx,
                'feature_name': feature_name,
                'baseline_accuracy': baseline_acc,
                'modified_accuracy': modified_acc,
                'accuracy_drop': accuracy_drop,
                'relative_importance': accuracy_drop / baseline_acc if baseline_acc > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def analyze_temporal_patterns(self, model, X_test, Y_test):
        """Analyze temporal importance patterns (early vs late packets)."""
        print("🔍 Analyzing temporal patterns...")
        
        baseline_pred = model.predict(X_test, batch_size=256, verbose=0)
        baseline_acc = self._calculate_accuracy(Y_test, baseline_pred)
        
        height = X_test.shape[1]
        results = []
        
        # Define temporal segments
        segments = {
            'early': (0, height // 3),
            'middle': (height // 3, 2 * height // 3),
            'late': (2 * height // 3, height)
        }
        
        for segment_name, (start, end) in segments.items():
            print(f"   Analyzing {segment_name} packets ({start}-{end})")
            
            # Zero out temporal segment
            X_modified = X_test.copy()
            X_modified[:, start:end, :] = 0
            
            # Evaluate
            modified_pred = model.predict(X_modified, batch_size=256, verbose=0)
            modified_acc = self._calculate_accuracy(Y_test, modified_pred)
            
            accuracy_drop = baseline_acc - modified_acc
            
            results.append({
                'segment': segment_name,
                'start_packet': start,
                'end_packet': end,
                'packets_count': end - start,
                'baseline_accuracy': baseline_acc,
                'modified_accuracy': modified_acc,
                'accuracy_drop': accuracy_drop,
                'relative_importance': accuracy_drop / baseline_acc if baseline_acc > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy for binary or multiclass predictions."""
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Multiclass
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=1)
        else:
            # Binary
            pred_classes = (y_pred > 0.5).astype(int).flatten()
            true_classes = y_true.flatten()
        
        return np.mean(pred_classes == true_classes)
    
    def create_visualizations(self, results_dict, output_dir):
        """Create visualizations for different analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        plt.style.use('seaborn-v0_8')
        
        # Individual feature/packet analysis
        if 'individual' in results_dict:
            self._plot_feature_heatmap(results_dict['individual'], output_dir)
        
        # Packet importance
        if 'packet' in results_dict:
            self._plot_packet_importance(results_dict['packet'], output_dir)
        
        # Feature type importance
        if 'feature_type' in results_dict:
            self._plot_feature_type_importance(results_dict['feature_type'], output_dir)
        
        # Temporal patterns
        if 'temporal' in results_dict:
            self._plot_temporal_patterns(results_dict['temporal'], output_dir)
    
    def _plot_feature_heatmap(self, df, output_dir):
        """Create heatmap of feature importance."""
        if df.empty:
            return
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot(index='packet_idx', columns='feature_idx', values='accuracy_drop')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   xticklabels=[self.feature_descriptions['cnn_features'].get(i, f'F{i}') for i in range(heatmap_data.shape[1])],
                   yticklabels=[f'Packet {i}' for i in range(heatmap_data.shape[0])])
        plt.title('Feature Importance Heatmap\n(Accuracy Drop when Feature is Zeroed)')
        plt.xlabel('Feature Type')
        plt.ylabel('Packet Position')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_packet_importance(self, df, output_dir):
        """Plot packet-level importance."""
        plt.figure(figsize=(12, 6))
        plt.bar(df['packet_idx'], df['accuracy_drop'])
        plt.xlabel('Packet Position in Time Window')
        plt.ylabel('Accuracy Drop')
        plt.title('Packet-Level Importance\n(Impact of Removing Entire Packets)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'packet_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_type_importance(self, df, output_dir):
        """Plot feature type importance."""
        plt.figure(figsize=(10, 6))
        plt.barh(df['feature_name'], df['accuracy_drop'])
        plt.xlabel('Accuracy Drop')
        plt.title('Feature Type Importance\n(Impact of Removing Feature Across All Packets)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_type_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_patterns(self, df, output_dir):
        """Plot temporal pattern importance."""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['segment'], df['accuracy_drop'])
        plt.xlabel('Temporal Segment')
        plt.ylabel('Accuracy Drop')
        plt.title('Temporal Pattern Importance\n(Impact of Removing Packet Segments)')
        
        # Add value labels on bars
        for bar, value in zip(bars, df['accuracy_drop']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results_dict, output_dir):
        """Generate a comprehensive analysis report."""
        report_path = Path(output_dir) / 'lucid_feature_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# LUCID Feature Analysis Report\n\n")
            f.write("This report analyzes feature importance in LUCID DDoS detection models using feature reset evaluation.\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            if 'feature_type' in results_dict:
                df = results_dict['feature_type']
                most_important = df.loc[df['accuracy_drop'].idxmax()]
                f.write(f"- **Most important feature type**: {most_important['feature_name']} "
                       f"(accuracy drops by {most_important['accuracy_drop']:.4f} when removed)\n")
                f.write(f"- **Baseline accuracy**: {most_important['baseline_accuracy']:.4f}\n")
            
            if 'temporal' in results_dict:
                df = results_dict['temporal']
                most_important_segment = df.loc[df['accuracy_drop'].idxmax()]
                f.write(f"- **Most important temporal segment**: {most_important_segment['segment']} packets "
                       f"(accuracy drops by {most_important_segment['accuracy_drop']:.4f})\n")
            
            if 'packet' in results_dict:
                df = results_dict['packet']
                avg_packet_importance = df['accuracy_drop'].mean()
                f.write(f"- **Average packet importance**: {avg_packet_importance:.4f} accuracy drop per packet\n")
            
            f.write("\n")
            
            # Detailed Analysis
            for analysis_type, df in results_dict.items():
                f.write(f"## {analysis_type.replace('_', ' ').title()} Analysis\n\n")
                
                if analysis_type == 'feature_type':
                    f.write("Impact of removing each feature type across all packets:\n\n")
                    f.write("| Feature Type | Accuracy Drop | Relative Importance |\n")
                    f.write("|--------------|---------------|--------------------|\n")
                    for _, row in df.iterrows():
                        f.write(f"| {row['feature_name']} | {row['accuracy_drop']:.4f} | {row['relative_importance']:.2%} |\n")
                
                elif analysis_type == 'temporal':
                    f.write("Impact of removing temporal segments:\n\n")
                    f.write("| Segment | Packets | Accuracy Drop | Relative Importance |\n")
                    f.write("|---------|---------|---------------|--------------------|\n")
                    for _, row in df.iterrows():
                        f.write(f"| {row['segment']} | {row['start_packet']}-{row['end_packet']} | {row['accuracy_drop']:.4f} | {row['relative_importance']:.2%} |\n")
                
                elif analysis_type == 'packet':
                    f.write("Per-packet importance statistics:\n\n")
                    f.write(f"- **Most important packet**: Position {df.loc[df['accuracy_drop'].idxmax(), 'packet_idx']} (drop: {df['accuracy_drop'].max():.4f})\n")
                    f.write(f"- **Least important packet**: Position {df.loc[df['accuracy_drop'].idxmin(), 'packet_idx']} (drop: {df['accuracy_drop'].min():.4f})\n")
                    f.write(f"- **Average importance**: {df['accuracy_drop'].mean():.4f}\n")
                    f.write(f"- **Standard deviation**: {df['accuracy_drop'].std():.4f}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the feature analysis:\n\n")
            
            if 'feature_type' in results_dict:
                df = results_dict['feature_type']
                high_impact = df[df['accuracy_drop'] > df['accuracy_drop'].mean() + df['accuracy_drop'].std()]
                low_impact = df[df['accuracy_drop'] < df['accuracy_drop'].mean() - df['accuracy_drop'].std()]
                
                f.write(f"1. **Critical Features**: Focus on {', '.join(high_impact['feature_name'])} as they have high impact\n")
                if not low_impact.empty:
                    f.write(f"2. **Optimization Potential**: Consider reducing emphasis on {', '.join(low_impact['feature_name'])}\n")
            
            f.write("3. **Robustness**: Test model performance under feature perturbation scenarios\n")
            f.write("4. **Adversarial Defense**: Protect high-impact features from adversarial manipulation\n")
            
        print(f"📄 Report generated: {report_path}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="LUCID Feature Analysis Tool")
    parser.add_argument('-m', '--model', required=True, help="Path to LUCID model (.h5)")
    parser.add_argument('-d', '--data', required=True, help="Path to test dataset (.hdf5)")
    parser.add_argument('-o', '--output', default='./lucid_analysis_results', help="Output directory")
    parser.add_argument('--max-samples', type=int, default=1000, help="Max samples to analyze")
    parser.add_argument('--max-packets', type=int, default=20, help="Max packets to analyze for CNN")
    parser.add_argument('--max-features', type=int, default=10, help="Max features to analyze for CNN")
    parser.add_argument('--quick', action='store_true', help="Quick analysis with reduced scope")
    
    args = parser.parse_args()
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        import h5py
        
        # Load model and data
        print(f"🚀 Loading model: {args.model}")
        model = load_model(args.model)
        
        print(f"📊 Loading data: {args.data}")
        with h5py.File(args.data, 'r') as f:
            X_test = f['x'][:args.max_samples] if 'x' in f else f['X'][:args.max_samples]
            Y_test = f['y'][:args.max_samples] if 'y' in f else f['Y'][:args.max_samples]
        
        # Ensure proper dimensions
        if len(X_test.shape) == 3 and len(model.input_shape) == 4:
            X_test = np.expand_dims(X_test, axis=-1)
        
        print(f"📈 Data shape: {X_test.shape}")
        print(f"🏷️  Labels shape: {Y_test.shape}")
        
        # Initialize analyzer
        analyzer = LucidFeatureAnalyzer()
        results = {}
        
        if len(X_test.shape) == 4:  # CNN model
            print("\n🧠 Detected CNN model - running comprehensive analysis...")
            
            # Feature type analysis (most important)
            results['feature_type'] = analyzer.analyze_feature_type_importance(model, X_test, Y_test)
            
            # Temporal pattern analysis
            results['temporal'] = analyzer.analyze_temporal_patterns(model, X_test, Y_test)
            
            # Packet-level analysis
            results['packet'] = analyzer.analyze_packet_importance(model, X_test, Y_test)
            
            # Individual feature analysis (if not quick mode)
            if not args.quick:
                results['individual'] = analyzer.analyze_cnn_features(
                    model, X_test, Y_test, 
                    max_packets=args.max_packets, 
                    max_features=args.max_features
                )
        
        else:
            print("\n🔢 Detected flatten/MLP model")
            print("   Note: Detailed CNN-style analysis not applicable to flatten models")
            # Could add specific flatten model analysis here
        
        # Create visualizations and report
        print("\n📊 Creating visualizations...")
        analyzer.create_visualizations(results, args.output)
        
        print("📄 Generating report...")
        analyzer.generate_report(results, args.output)
        
        # Save numerical results
        for analysis_type, df in results.items():
            df.to_csv(Path(args.output) / f'{analysis_type}_analysis.csv', index=False)
        
        print(f"\n✅ Analysis complete! Results saved in: {args.output}")
        
        # Print key findings
        print(f"\n🔍 Key Findings:")
        if 'feature_type' in results:
            df = results['feature_type']
            top_feature = df.loc[df['accuracy_drop'].idxmax()]
            print(f"   • Most critical feature: {top_feature['feature_name']} (impact: {top_feature['accuracy_drop']:.4f})")
        
        if 'temporal' in results:
            df = results['temporal']
            top_segment = df.loc[df['accuracy_drop'].idxmax()]
            print(f"   • Most critical time segment: {top_segment['segment']} (impact: {top_segment['accuracy_drop']:.4f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
