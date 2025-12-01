#!/usr/bin/env python3
"""
File: visualize_results.py
Project: SPLADE-PT-BR
Created: Monday, 1st December 2025
Author: Axel

Purpose

  Visualize SPLADE-PT-BR training and evaluation results.

  This script creates visualizations for:
    - Training loss curve
    - Sparsity distribution
    - Metrics comparison charts

Copyright (c) 2025 Axel. All rights reserved.

"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Paths (script is in scripts/utils/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLADE_DIR = PROJECT_ROOT / "splade"
CHECKPOINT_DIR = SPLADE_DIR / "experiments" / "pt" / "checkpoint"
RESULTS_DIR = SPLADE_DIR / "experiments" / "pt" / "out"
OUTPUT_DIR = RESULTS_DIR / "plots"

# Files
TRAINING_PERF_FILE = CHECKPOINT_DIR / "training_perf.txt"
COMPARISON_FILE = RESULTS_DIR / "comparison_report.json"


def setup_plot_style():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_training_loss():
    """Plot training loss curve"""
    print("📈 Creating training loss plot...")
    
    if not TRAINING_PERF_FILE.exists():
        print(f"   ⚠️  Training performance file not found: {TRAINING_PERF_FILE}")
        return
    
    # Read training data
    iterations = []
    losses = []
    
    with open(TRAINING_PERF_FILE, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) == 2:
                    try:
                        iter_num = int(parts[0])
                        loss = float(parts[1])
                        iterations.append(iter_num)
                        losses.append(loss)
                    except ValueError:
                        continue
    
    if not iterations:
        print("   ⚠️  No training data found")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Full training curve
    ax1.plot(iterations, losses, linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Batch Ranking Loss')
    ax1.set_title('Training Loss Curve (Full)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Last 50k iterations (zoomed)
    if len(iterations) > 50:
        start_idx = max(0, len(iterations) - 50)
        ax2.plot(iterations[start_idx:], losses[start_idx:], 
                linewidth=2, color='#A23B72', alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Batch Ranking Loss')
        ax2.set_title('Training Loss Curve (Last Iterations)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / "training_loss.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to: {output_file}")
    
    # Print statistics
    print(f"\n   📊 Training Statistics:")
    print(f"      Total iterations: {iterations[-1]:,}")
    print(f"      Initial loss: {losses[0]:.6f}")
    print(f"      Final loss: {losses[-1]:.6f}")
    print(f"      Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")


def plot_metrics_comparison():
    """Plot metrics comparison with original SPLADE"""
    print("\n📊 Creating metrics comparison chart...")
    
    if not COMPARISON_FILE.exists():
        print(f"   ⚠️  Comparison file not found: {COMPARISON_FILE}")
        print("   Run scripts/utils/compare_models.py first")
        return
    
    with open(COMPARISON_FILE, 'r') as f:
        comparison = json.load(f)
    
    if "comparison" not in comparison or not comparison["comparison"]:
        print("   ⚠️  No comparison data available")
        return
    
    # Extract metrics
    metrics = []
    original_values = []
    pt_values = []
    
    for metric_name, data in comparison["comparison"].items():
        if data["splade_pt_br"] != "N/A":
            metrics.append(metric_name)
            original_values.append(data["original"])
            pt_values.append(data["splade_pt_br"])
    
    if not metrics:
        print("   ⚠️  No comparable metrics found")
        return
    
    # Create plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, original_values, width, label='SPLADE++ (Original - EN)', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, pt_values, width, label='SPLADE-PT-BR (PT)', 
                   color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('SPLADE-PT-BR vs Original SPLADE++ Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / "metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to: {output_file}")


def plot_sparsity_info():
    """Create informational chart about sparsity"""
    print("\n🎯 Creating sparsity information chart...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    categories = ['Total\nVocabulary', 'Active Dims\n(Query)', 'Active Dims\n(Document)']
    values = [29794, 120, 150]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label = f'{value:,}' if value > 1000 else str(value)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label,
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Dimensions')
    ax.set_title('SPLADE-PT-BR Sparsity: ~99.5% Sparse Vectors')
    ax.set_ylim(0, 32000)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sparsity percentage
    query_sparsity = (1 - 120/29794) * 100
    doc_sparsity = (1 - 150/29794) * 100
    
    textstr = f'Query Sparsity: {query_sparsity:.2f}%\nDocument Sparsity: {doc_sparsity:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / "sparsity_info.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to: {output_file}")


def main():
    """Main visualization workflow"""
    print("=" * 80)
    print("📊 SPLADE-PT-BR Results Visualization")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup matplotlib style
    setup_plot_style()
    
    # Create plots
    plot_training_loss()
    plot_sparsity_info()
    plot_metrics_comparison()
    
    print("\n" + "=" * 80)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    # List generated files
    print("\n📁 Generated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   • {file.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

